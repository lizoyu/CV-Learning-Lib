/*************************************************************************
	> File Name: 6.1_alter2.cpp
	> Author: 
	> Mail: 
	> Created Time: 2017年02月01日 星期三 08时21分16秒
 ************************************************************************/

#include<iostream>
#include<Eigen/Dense>
using namespace std;
using namespace Eigen;

VectorXd GeneClassifierNorm( MatrixXd trainData, VectorXd state,
                            int num_class, double point)
{
    double K = num_class;
    for( int k = 0; k < K; ++k )
    {
        // set mean(temp)
        double sum_delta;
        for( int i = 0; i < state.size(); ++i )
        {
            if( state(i) == k + 1 )
                sum_delta++;
        }
        VectorXd mean;
        for( int i = 0; i < state.size(); ++i )
        {
            if( state(i) == k + 1 )
                mean += trainData.col( i );
        }
        mean = mean / sum_delta;

        // set variance(temp)
        MatrixXd var;
        // set prior(save)

        // likelihoods(save)
    
    }
    // classify the new datapoint
    for( int k = 0; k< K; ++k )
    {
        
    }
    return 0;
}
