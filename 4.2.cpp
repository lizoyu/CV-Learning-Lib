/*************************************************************************
	> File Name: 4.2.cpp - MAP learning of normal with conjugate prior
	> Author: Chungyuen Li
	> Mail: lizoyu1@gmail.com
	> Created Time: 2017年01月19日 星期四 22时18分39秒
 ************************************************************************/

#include<iostream>
#include<cmath>
#include<vector>
using namespace std;
using std::vector;

vector<double> MAPforNormal ( vector<double>::iterator begin,
                            vector<double>::iterator end,
                            double alpha, double beta,
                             double gamma,  double delta)
{
    double mean, var, sum, size, sum_devi;

    // calculate the mean
    for( vector<double>::iterator iter = begin;
        iter != end; ++iter )
    {
        sum += *iter;
        size++;
    }
    mean = ( sum + gamma*delta ) / ( size + gamma );

    // calculate the variance
    for( vector<double>::iterator iter = begin;
       iter != end; ++iter )
        sum_devi += pow( *iter - mean, 2 );
    var = ( sum_devi + 2*beta + gamma*pow( delta - mean, 2 ) ) / ( size + 3 + 2*alpha );

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
