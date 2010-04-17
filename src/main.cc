#include <iostream>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <sstream>
#include <iterator>
#include <cstdlib>
#include <cmath>

#include "matrix.h"
#include "gaussian_elimination.h"
#include "parallel_jacobi.h"
#include "timer.h"

typedef std::vector<float> float_vec;

void generate_symmetric_matrix(matrix* A)
{
	// Fill lower triangle and diagonal of matrix
	for(int i=0;i<A->size();++i)
		for(int j=0;j<=i;++j)
			A->get(i,j)=(static_cast<float>(rand())/RAND_MAX - 0.5f)*10.0f;
	// Copy from lower triangle
	for(int i=0;i<A->size();++i)
		for(int j=i+1;j<A->size();++j)
			A->get(i,j) = A->get(j,i);
}

void init_random(matrix** mat, int n)
{
	*mat = new matrix(n);
	srand(0);
	generate_symmetric_matrix(*mat);
	std::cout << "Generated "<<n<<"*"<<n<<" matrix from random numbers\n";
}

void read_stdin(matrix** mat)
{
	unsigned int size;
	std::cin >> size;
	*mat = new matrix(size);

	for(int i=0;i<size*size;++i)
		std::cin >> (*mat)->get(i/size,i%size);
}

void print_timing_stats(timer& t, int matsize)
{
#ifdef enable_openmp
	// if parallel_jacobi then load serial timings to calculate speedup
	timer* ser=0;
	std::ifstream stimes("timings/serial_timers.txt");
	if(stimes.is_open())
		stimes >> &ser;
	stimes.close();
	
	double efactor = 100.0/static_cast<double>(omp_get_max_threads());
	double speeduppre = ser->get("pre-multiplication")
		/ t.get("pre-multiplication");
	double speeduppost = ser->get("post-multiplication")
		/ t.get("post-multiplication");
	double speeduptotal = ser->get("run")/t.get("run");
	double efficiencypre = speeduppre * efactor;
	double efficiencypost = speeduppost * efactor;
	double efficiencytotal = speeduptotal * efactor;
	
	std::cout << "Parallel speedup:\n"
		<< "  pre-multiplication: " << speeduppre << "\n"
		<< "  post-multiplication: " << speeduppost << "\n"
		<< "  total: " << speeduptotal << "\n\n";

	std::cout << "Parallel efficiency:\n"
		<< "  pre-multiplication: " << efficiencypre << "%\n"
		<< "  post-multiplication: " << efficiencypost << "%\n"
		<< "  total: " << efficiencytotal << "%\n\n";
	
	// Write speedup and elapsed times to files for graphing.
	t.print_threads_v_elapsed();
	t.print_threads_v_speedup(ser);
	//t.print_threads_v_efficiency(ser);
	
	std::ostringstream oss;
	oss << "timings/efficiency" << matsize << ".txt";
	std::ofstream efflog(oss.str().c_str(), std::ios::app);
	efflog << omp_get_max_threads() << " " << efficiencytotal << "\n";
	efflog.close();
#else
	// Write elapsed times to a file for reading in the parallel version.
	std::ofstream tlog("timings/serial_timers.txt");
	tlog << t;
	tlog.close();
	
	// Write elapsed times to graphing files.
	t.print_threads_v_elapsed();
#endif
}

int print_usage()
{
	std::cout << "Usage: parallel_jacobi mode [options]\n"
		<< "Where mode can be one of the following:\n"
		<< "\tthreshold T - Terminate iterations when the off-diagonal magnitude becomes less than T.\n"
		<< "\titerations I - Terminate after I iterations.\n"
		<< "\tdifference D - Terminate when the difference between of the off-diagonal magnitude between consecutive iterations is less than D.\n"
		<< "Other options:\n"
		<< "\t--random N - Generate an N*N symmetric matrix to use.\n"
		<< "\t--check - Verify each calculated eigenvalue is accurate.\n"
		<< "\t--quiet - Don't print the eigenvalues found.\n"
		<< "Input: Either use --random N, or read from standard input:\nFirst line: size of matrix, N. Next N lines have N floats describing the matrix.\n"
		<< "parallel_jacobi was written by Simon Loach\n";
	return 1;
}

enum mode { threshold, iterations, difference };

int main(int argc, const char* argv[])
{
	matrix* A=0;
	matrix* original=0;
	
	// Command line options
	bool checkSingular = false;
	bool printEigenvalues = true;
	mode currentMode = threshold;
	int matSizeRandom = 0;
	unsigned int numIterations = 0;
	double offThreshold = 10e-6;
	double offDifference = 10e-2;
	
	// Parse command line
	if(argc < 2) return print_usage();
	for(int i=1;i<argc;++i)
	{
		std::string opt(argv[i]);
		if(opt == "--check") checkSingular = true;
		else if(opt == "--quiet") printEigenvalues = false;
		else if(opt == "--mode")
		{
			// Ensure there are enough arguments passed.
			if(i+2 >= argc) return print_usage();
			
			std::string modeopt(argv[++i]);
			if(modeopt == "threshold")
			{
				currentMode = threshold;
				offThreshold = atof(argv[++i]);
			}
			else if(modeopt == "iterations")
			{
				currentMode = iterations;
				numIterations = atoi(argv[++i]);
			}
			else if(modeopt == "difference")
			{
				currentMode = threshold;
				offDifference = atof(argv[++i]);
			}
		}
		else if(opt == "--random")
		{
			if(i+1 >= argc) return print_usage();
			matSizeRandom = atoi(argv[++i]);
		}
		else
		{
			std::cerr << "Unknown option " << opt << std::endl;
			return print_usage();
		}
	}
	
	// Read in matrix or generate one if required
	if(matSizeRandom > 0)
		init_random(&A, matSizeRandom);
	else
		read_stdin(&A);

	// Store a copy of the original matrix for use checking eigenvalues, if
	// required.
	if(checkSingular) original = new matrix(*A);

	// Print information about runtime.
#ifdef enable_openmp
	std::cout << "\nRunning parallel jacobi on " << omp_get_max_threads()
		<< " threads.\n";
#else
	std::cout << "\nRunning serial jacobi.\n";	
#endif
	
	// Choose appropriate function to call depending on "mode" setting.
	timer timerroot("run");
	float_vec eigenvalues;
	if(currentMode == threshold)
	{
		parallel_jacobi::converge_off_threshold sc(offThreshold, *A);
		parallel_jacobi::music_permutation pe(A->size());
		parallel_jacobi::run(*A, sc, pe, timerroot);
	}
	else if(currentMode == iterations)
	{
		parallel_jacobi::converge_max_iterations sc(numIterations);
		parallel_jacobi::music_permutation pe(A->size());
		parallel_jacobi::run(*A, sc, pe, timerroot);
	}
	else if(currentMode == difference)
	{
		parallel_jacobi::converge_off_difference sc(offDifference);
		parallel_jacobi::music_permutation pe(A->size());
		parallel_jacobi::run(*A, sc, pe, timerroot);
	}
	
	std::cout << "\nTiming statistics\n";
	timerroot.print(std::cout, 0, 0.0);
	std::cout << "\n";
	print_timing_stats(timerroot, A->actual_size());
	
	// Retrieve the eigenvalues from the diagonal
	for(int i=0;i<A->actual_size();++i)
		eigenvalues.push_back(A->get(i,i));
		
	// Finished with this copy of A
	delete A;
	A=0;
	
	if(printEigenvalues)
	{
		std::cout << "Eigenvalues are: ";
		std::sort(eigenvalues.begin(), eigenvalues.end());
		for(float_vec::iterator it = eigenvalues.begin();
			it != eigenvalues.end();
			++it)
		{
			std::cout << std::setprecision(6) << (*it) << "; ";
		}
		std::cout << std::endl;
	}
	
	if(checkSingular)
	{
		std::cout << "\nVerifying eigenvalues using Gaussian elimination\n";
		// Calculate A-lambda*I for each eigenvalue lambda then use gaussian
		// elimination on it.
		for(int eig=0;eig<eigenvalues.size();++eig)
		{
			matrix aminuslambdai(*original);
			for(int i=0; i<aminuslambdai.actual_size();++i)
			{
				aminuslambdai.get(i,i) -= eigenvalues[eig];
			}
			bool s = gaussian_elimination(aminuslambdai);
			std::cout << "The matrix given by A-" << std::fixed << std::setprecision(4)
				<< eigenvalues[eig] << "*I is "
				<< (s?"invertible":"singular") << std::endl;
			if(s) std::cout << aminuslambdai;
		}
	}

	return 0;
}

