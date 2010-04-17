#include <boost/test/auto_unit_test.hpp>

// Boost Test declaration and Checking macros
#include <boost/test/unit_test_suite.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <vector>

#include "../gaussian_elimination.cc"
#include "singular_matrix_256.h"

matrix make_matrix(int n, double values[]) {
	matrix mat(n);
	for(int i=0;i<n;++i)
		for(int j=0;j<n;++j)
			mat(i,j) = values[i*n+j];
	return mat;
}

BOOST_AUTO_TEST_SUITE(gaussian_elimination_unittest)

	static const double epsilon = 1e-4;

	double mat1[] = { 0.0, 0.0, 0.0, 0.0 };
	double mat2[] = { 0.0, 0.0, 1.0, 1.0 };
	double mat3[] = { 1.0, 1.0, 1.0, 1.0 };
	double mat4[] = { 3.0, 2.0, 6.0, 4.0 };
	
	BOOST_AUTO_TEST_CASE(simple_1)
	{
		matrix m = make_matrix(2,mat1);
		BOOST_CHECK(gaussian_elimination(m) == false);
	}
	
	BOOST_AUTO_TEST_CASE(simple_2)
	{
		matrix m = make_matrix(2,mat2);
		BOOST_CHECK(gaussian_elimination(m) == false);
	}
	
	BOOST_AUTO_TEST_CASE(simple_3)
	{
		matrix m = make_matrix(2,mat3);
		BOOST_CHECK(gaussian_elimination(m) == false);
	}
	
	BOOST_AUTO_TEST_CASE(simple_4)
	{
		matrix m = make_matrix(2,mat4);
		BOOST_CHECK(gaussian_elimination(m) == false);
	}
	
	double mat5[] = { 1.0f, 0.0f, 0.0f, 1.0f };
	double mat6[] = { 2.0f, 3.0f, 4.0f, 5.0f };

	BOOST_AUTO_TEST_CASE(simple_5)
	{
		matrix m = make_matrix(2,mat5);
		BOOST_CHECK(gaussian_elimination(m) == true);
	}
	
	BOOST_AUTO_TEST_CASE(simple_6)
	{
		matrix m = make_matrix(2,mat6);
		BOOST_CHECK(gaussian_elimination(m) == true);
	}
	
	
	double mat7[] = {3.4019, -1.0562,  2.9844, -1.6478, -0.2260,  4.1620, -2.5711,  4.9892, -0.0642, -1.4754, -1.5111, -1.2479,  1.3998,  0.8864,  3.8106, -3.9683, -1.0562,  2.8310,  4.1165,  2.6823,  1.2887,  1.3571, -3.6277, -2.8174,  4.7278,  3.0772, -4.3583,  2.6025, -1.4595,  1.5730,  1.4108, -3.7392, 2.9844,  4.1165, -3.0245, -2.2223, -1.3522,  2.1730,  3.0418,  0.1293, -2.0748,  4.1903, -4.7998,  0.1254,  1.8786,  3.5868, -0.6805, -0.0456, 
	-1.6478,  2.6823, -2.2223,  0.5397,  0.1340, -3.5840, -3.4332,  3.3911,  2.7136, -4.3024, -0.4230,  1.6772, -3.3403, -0.6044,  1.1960,  2.6048, -0.2260,  1.2887, -1.3522,  0.1340,  4.5223,  1.0697, -0.9906,  1.1264,  0.2674,  4.4933, -4.3690,  0.3161, -0.5990,  4.2397, -2.1894,  4.8475,  4.1620,  1.3571,  2.1730, -3.5840,  1.0697, -4.8370, -3.7021, -2.0397,  2.6991,  0.2600, -2.6172, -4.6072,  3.8008, -1.0156,  2.8600,  4.3500, 
	-2.5711, -3.6277,  3.0418, -3.4332, -0.9906, -3.7021, -3.9119,  1.3755, -0.9977, -4.1394,  4.7063, -0.6236,  3.2920,  3.1477, -1.9254,  1.8445,  4.9892, -2.8174,  0.1293,  3.3911,  1.1264, -2.0397,  1.3755,  0.2429,  3.9153, -3.0779,  4.0221,  4.3184, -1.6966,  1.8422, -0.5297, -1.1681, -0.0642,  4.7278, -2.0748,  2.7136,  0.2674,  2.6991, -0.9977,  3.9153, -2.1669,  1.6323,  3.5092,  4.3081, -2.7103,  4.1097, -2.7389,  2.4977, 
	-1.4754,  3.0772,  4.1903, -4.3024,  4.4933,  0.2600, -4.1394, -3.0779,  1.6323,  3.9023, -2.3333,  2.2095,  3.9337, -0.1751, -3.1247, -1.3134, -1.5111, -4.3583, -4.7998, -0.4230, -4.3690, -2.6172,  4.7063,  4.0221,  3.5092, -2.3333,  0.3976, -2.1571, -1.4964, -2.8418, -2.2377, -2.0584, -1.2479,  2.6025,  0.1254,  1.6772,  0.3161, -4.6072, -0.6236,  4.3184,  4.3081,  2.2095, -2.1571,  2.3853,  1.8667,  4.5025,  0.5644, -2.6774, 
	 1.3998, -1.4595,  1.8786, -3.3403, -0.5990,  3.8008,  3.2920, -1.6966, -2.7103,  3.9337, -1.4964,  1.8667,  4.5647,  4.2013, -0.8350,  0.8449, 0.8864,  1.5730,  3.5868, -0.6044,  4.2397, -1.0156,  3.1477,  1.8422,  4.1097, -0.1751, -2.8418,  4.5025,  4.2013, -3.5234, -3.3039, -2.5559, 3.8106,  1.4108, -0.6805,  1.1960, -2.1894,  2.8600, -1.9254, -0.5297, -2.7389, -3.1247, -2.2377,  0.5644, -0.8350, -3.3039,  4.0680, -3.4761, 
	-3.9683, -3.7392, -0.0456,  2.6048,  4.8475,  4.3500,  1.8445, -1.1681,  2.4977, -1.3134, -2.0584, -2.6774,  0.8449, -2.5559, -3.4761,  2.3215 };
	
	BOOST_AUTO_TEST_CASE(large)
	{
		matrix m = make_matrix(16,mat7);
		BOOST_CHECK(gaussian_elimination(m) == true);
	}
	
	BOOST_AUTO_TEST_CASE(verylarge)
	{
		matrix m = make_matrix(256,singular256);
		BOOST_CHECK(gaussian_elimination(m) == false);
	}
	

BOOST_AUTO_TEST_SUITE_END()

