#include "gaussian_elimination.h"
#include <cmath>
#include <iostream>
using std::fabs;

// Returns true if Gaussian elimination succeeds.
bool gaussian_elimination(matrix& A)
{
	const int n=A.actual_size();
	const float epsilon=1e-2f * frobenius_norm(A);

	for(int k=0; k<n; ++k)
	{
		// Determine for this column the largest entry mu.
		int mu = k;
		for(int i=k; i<n; ++i)
			if(fabs(A(i,k)) > fabs(A(mu,k))) mu=i;
		
		A.swap_rows(k,mu);
		
		const float pivot = A(k,k);
		// If pivot is zero then matrix is linearly dependent and hence singular
		if(fabs(pivot) > epsilon)
		{
			// Subtract scalar multiples of the pivot row from the rest
			// to get zeros in the pivot column
			for(int r=k+1; r<n; ++r)
			{
				float multiple = A(r,k)/pivot;
				for(int u=k;u<n;++u)
					A(r,u) = A(r,u) - multiple*A(k,u);
			}
		}
		else // zero row found
			return false;
	}
	return true;
}
