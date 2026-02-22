import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh
import argparse
import sys

class Hamiltonian2D:
	def __init__(self, N=20, potential='well'):
		self.N = N
		self.potential = potential
		self.dx = 1.0 / float(N) # grid spacing, can be arbitrary
		self.inv_dx2 = float(N * N) # 1/dx^2

	# Helper function to map (i,j) -> linear index
	def idx(self, i, j):
		return i * self.N + j
			
	# Potential function
	def V(self, i, j):
		# Example 1: infinite square well -> zero in interior, large outside
		if self.potential == 'well':
			# No boundary enforcement here, but can skip boundary wavefunction
			return 0.
		# Example 2: 2D harmonic oscillator around center
		elif self.potential == 'harmonic':
			x = (i - self.N/2) * self.dx
			y = (j - self.N/2) * self.dx
			# Quadratic potential V = k * (x^2 + y^2)
			return 4. * (x**2 + y**2)
		elif self.potential == 'double_well':
			# Center coordinates around 0
			x = (i - self.N/2) * self.dx
			y = (j - self.N/2) * self.dx
            
            # Quartic Double Well in x: V(x) = A*(x^2 - a^2)^2
            # Harmonic in y: V(y) = B*y^2
            # The minima are at x = +0.2 and x = -0.2
			A = 50000.0  # Barrier height modifier
			a = 0.3     # Distance of wells from center
			B = 50.0    # Squeeze in y-direction
			return A * (x**2 - a**2)**2 + B * y**2
		else:
			return 0.

	def build_2d_hamiltonian(self):
		"""
		Build a discretized 2D Hamiltonian on an N x N grid.
		
		Parameters
		----------
		N : int
			Number of points in each dimension (N^2 total points).
		potential : str
			Choose the potential. 'well' or 'harmonic' examples.
		Returns
		-------
		H : ndarray of shape (N^2, N^2)
			The Hamiltonian matrix approximating -d^2/dx^2 - d^2/dy^2 + V(x,y).
		"""
		H = lil_matrix((self.N*self.N, self.N*self.N), dtype=np.float64)
				
		# Build the matrix: For each (i, j), set diagonal for 2D Laplacian plus V
		for i in range(self.N):
			for j in range(self.N):
				row = self.idx(i,j)
				# Potential
				H[row, row] = 4. * self.inv_dx2 + self.V(i,j) # "Kinetic" ~ -4/dx^2 in 2D FD
				# Neighbors (assuming no boundary conditions or Dirichlet)
				if i > 0: # up
					H[row, self.idx(i-1, j)] = -self.inv_dx2
				if i < self.N-1: # down
					H[row, self.idx(i+1, j)] = -self.inv_dx2
				if j > 0: # left
					H[row, self.idx(i, j-1)] = -self.inv_dx2
				if j < self.N-1: # right
					H[row, self.idx(i, j+1)] = -self.inv_dx2
					
		return H.tocsr() # Compressed Sparse Row format

	def build_2d_hamiltonian_generalized(self, a=0., b=0.):
		M = self.N - 2
		H = lil_matrix((M**2, M**2), dtype=np.float64)

		def idx(i, j): 
			return (i-1) * M + (j-1)

		def boundary_val(i, j):
			x = i * self.dx
			y = j * self.dx
			return a * x + b * y

		for i in range(1,self.N - 1):
			for j in range(1,self.N - 1):
				row = idx(i, j)
				H[row, row] = 4. * self.inv_dx2 + self.V(i, j)
				# interior neighbors
				for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
					ni, nj = i+di, j+dj
					if 1 <= ni <=self.N-2 and 1 <= nj <=self.N-2:
						H[row, idx(ni, nj)] = -self.inv_dx2
					else:
						# boundary neighbor: shift diagonal
						H[row, row] += self.inv_dx2 * boundary_val(ni, nj)
		return H.tocsr()
	
	def solve_eigen(self,n_eigs=5):
		"""
		Build a 2D Hamiltonian and solve for the lowest n_eigs eigenvalues.
		
		Parameters
		----------
		N : int
			Grid points in each dimension.
		potential : str
			Potential type.
		n_eigs : int
			Number of eigenvalues to return.
		
		Returns
		-------
		vals : array_like
			The lowest n_eigs eigenvalues sorted ascending.
		vecs : array_like
			The corresponding eigenvectors.
		"""

		H = self.build_2d_hamiltonian()
		#H = self.build_2d_hamiltonian_generalized(a=0,b=0)
		# Solve entire spectrum (careful for large N)
		vals, vecs = eigsh(H, k=n_eigs, which='SM') # smallest
		return vals, vecs

	def get_density(self,vecs):
		N = np.sqrt(len(vecs)).astype(int)
		psi_ground = vecs[:, 0].reshape((N, N))
		return np.abs(psi_ground)**2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="solve for the eigenvalues of a 2D hamiltonian"
    )

    # Adding arguments
    parser.add_argument(
        "--N", type=int, default=10,
        help="number of grid cells -- default: 10"
    )
    parser.add_argument(
        "--potential", type=str, default="well", choices=["well", "harmonic", "double_well"],
        help="type of potential to use -- default: well"
    )
    parser.add_argument(
        "--n-eigs", type=int, default=5,
        help="number of eigenvalues to return -- default: 5"
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="location to save results -- default: ./eigs_N{N}.txt"
    )
    parser.add_argument('--prob', action='store_true', help='Save probability density of ground state')

    args = parser.parse_args()
	
    if args.out is None:
        args.out = f"eigs_N{args.N}.txt"

    if args.N <= 0:
        print(f"error: N must be a positive integer. input: {args.N}")
        sys.exit(1)
    
    if args.n_eigs <= 0:
        print("error: n-eigs must be a positive integer.")
        sys.exit(1)

    if args.n_eigs > args.N**2:
        print(f"error: requested {args.n_eigs} eigenvalues, but max eigenvalues is N^2, {args.N**2}")
        sys.exit(1)

    print(f"solving 2D hamiltonian with N={args.N}, and potential={args.potential}")
    solver = Hamiltonian2D(N=args.N, potential=args.potential)
    vals, vecs = solver.solve_eigen(n_eigs=args.n_eigs)
    np.savetxt(f'{args.out}', vals)
    print(f"Saved lowest {len(vals)} eigenvalues to output file: {args.out}")
    if args.prob:
        np.savetxt(f'density_N{args.N}_{args.potential}.txt', solver.get_density(vecs))