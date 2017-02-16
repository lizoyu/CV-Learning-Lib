/*************************************************************************
	> File Name: 6.1.cpp
	> Author: 
	> Mail: 
	> Created Time: 2017年02月01日 星期三 08时21分16秒
 ************************************************************************/

#include<iostream>
#include<Eigen/Dense>
using namespace std;
using namespace Eigen;

VectorXd GeneClassifierNorm( MatrixXd trainData, VectorXd state,
                            int num_class, VectorXd point)
{
    int K = num_class;
    VectorXd lambda(K), l(K);
    for( int k = 0; k < K; ++k )
    {
        // set mean(temp)
        double sum_delta = 0;
        for( int i = 0; i < state.size(); ++i )
        {
            if( state(i) == k + 1 )
                sum_delta++;
        }
        VectorXd mean(trainData.rows());
        mean.setZero();
        for( int i = 0; i < state.size(); ++i )
        {
            if( state(i) == k + 1 )
                mean += trainData.col( i );
        }
        mean = mean / sum_delta;

        // set variance(temp)
        MatrixXd var(trainData.rows(), trainData.rows());
        var.setZero();
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
        lambda(k) = sum_delta / state.size();

        // likelihoods(save)
        VectorXd deviation = point - mean;
        MatrixXd var_inv = var.jacobiSvd(ComputeThinU|ComputeThinV).solve(
            MatrixXd::Identity(var.rows(),var.cols())
            );
        l(k) = exp( -0.5*deviation.transpose()*var_inv*deviation) /
        ( pow( 2*M_PI, round(trainData.rows()/2) )*sqrt(
            abs(var.determinant() ) ));
    }
    // classify the new datapoint
    double sum_lk;
    for( int i = 0; i < lambda.size(); ++i )
        sum_lk += l(i) * lambda(i);
    cout << "Sum_lk:" << endl << sum_lk << endl;

    VectorXd P(K);
    for( int k = 0; k< K; ++k )
        P(k) = l(k)*lambda(k) / sum_lk;
    
    return P;
}

int main()
{
    // two class, 1 - (10, 20, 30); 2 - (70, 80, 90);
    int num_class = 2;
    MatrixXd trainData(3,6);
    trainData << 1.3, 0.9, 1.1, 7.4, 6.8, 7.1,
                 2.2, 1.8, 2.1, 8.2, 7.6, 8.1,
                 3.1, 2.4, 3.1, 9.1, 8.9, 9.1;
    trainData *= 10;
    VectorXd state(6);
    state << 1, 1, 1, 2, 2, 2;
    VectorXd point(3);
    point << 6.8, 7.8, 8.8;
    point *= 10;

    VectorXd P = GeneClassifierNorm( trainData, state, num_class, point );
    cout << "P:" << endl << P << endl;
    return 0;
}
