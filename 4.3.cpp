/*************************************************************************
	> File Name: 4.3.cpp - Bayesian learning for normal
	> Author: Chungyuen Li
	> Mail: lizoyu1@gmail.com
	> Created Time: 2017年01月19日 星期四 22时18分39秒
 ************************************************************************/

#include<iostream>
#include<cmath>
#include<vector>
using namespace std;
using std::vector;

vector<double> BayeforNormal ( vector<double>::iterator begin,
                            vector<double>::iterator end,
                            double alpha, double beta,
                             double gamma,  double delta, double testPoint;)
{
    double mean, var, sum, size, sum_2;

    // calculate the sum and size
    for( vector<double>::iterator iter = begin;
        iter != end; ++iter )
    {
        sum += *iter;
        sum_2 += pow( *iter, 2 );
        size++;
    }

    // calculate the posterior over normal
    alpha_1 = alpha + size/2;
    beta_1 = sum_2/2 + beta + gamma*pow( delta, 2 )/2 - pow( gamma*delta + sum, 2 ) / ( 2*gamma + 2*size );
    gamma_1 = gamma + size;
    delta_1 = ( gamma*delta + sum ) / ( gamma + size );

    // calculate the intermediate parameters
    alpha_2 = alpha_1 + 0.5;
    beta_2 = pow( testPoint, 2 )/2 + beta_1 + gamma_1*pow( delta_1, 2 )/2 - pow( gamma_1*delta_1 + testPoint, 2 ) / ( 2*gamma_1 + 2 );
    gamma_2 = gamma_1 + 1;

    // evaluate the new datapoint
    double P = 
    
    // save the mean and variance in vector "result"
    vector<double> result(2);
    result[0] = mean;
    result[1] = var;
    return result;
}


// testing
int main()
{
    vector<double> trainData;
    for( int i = 0; i < 11; ++i )
        trainData.push_back(i);
    vector<double> result = MAPforNormal( trainData.begin(),
                                        trainData.end(),
                                        1, 1, 1, 0);
    cout << result[0] << ", " << result[1] << endl;

    return 0;
}
