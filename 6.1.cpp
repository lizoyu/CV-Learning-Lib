/*************************************************************************
	> File Name: 6.1_alter2.cpp
	> Author: 
	> Mail: 
	> Created Time: 2017年02月01日 星期三 08时21分16秒
 ************************************************************************/

#include<iostream>
#include<Eigen/Dense>
#include<Eigen/LU>
using namespace std;
using namespace Eigen;

VectorXd GeneClassifierNorm( MatrixXd trainData, VectorXd state,
                            int num_class, double point)
{
    double K = num_class;
    VectorXd lambda(K), l(K);
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
        for( int i = 0; i < state.size(); ++i )
        {
            if( state(i) == k + 1 )
            {
                VectorXd deviation = trainData.col( i ) - mean;
                var += deviation * deviation.transpose();
            }
        }
        var = var / sum_delta;

        // set prior(save)
        lambda(k) = sum_delta / trainData.cols();

        // likelihoods(save)
        VectorXd deviation = x - mean;
        l(k) = exp( -0.5*deviation.transpose()*var.inverse()*deviation) /
        ( pow( 2*M_PI, round(trainData.rows/2) )*sqrt( var.determinant() ) );
    }
    // classify the new datapoint
    double sum_lk;
    for( int i = 0; i < lambda.size(); ++i )
        sum_lk += l(i) * lambda(i);

    VectorXd P(K);
    for( int k = 0; k< K; ++k )
        P(k) = l(k)*lambda(k) / sum_lk;
    
    return P;
}

int main()
{
    
    return 0;
}
