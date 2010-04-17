#pragma once
#include "matrix.h"

// Modified version of Golub and Van Loan p112. Algorithm 3.4.1
// Returns true if the matrix A is singular.
bool gaussian_elimination(matrix& A);
