/*************************************************************************
	> File Name: 6.1_alter.cpp
	> Author: 
	> Mail: 
	> Created Time: 2017年01月27日 星期五 09时00分19秒
 ************************************************************************/

#include<iostream>
#include<vector>
using namespace std;

vector<double> GeneClassifierNorm( const vector<vector<double>> &trainData,
                                  const vector<vector<double>> &worldState,
                                  double num_class, double x)
{
    double num_variable = trainData[0].size();
    
    // sum of delta 
    vector<double> stat( num_class );
    for( vector<double>::iterator iter = trainData.begin();
       iter != trainData.end(); ++iter )
        stat[*iter - 1]++;

    // set mean
    vector<vector<double>> mean( num_class,
                                vector<double>(num_variable,0));
    for( int k = 1; k < mean.size() + 1; ++k )
    {
        for( int w = 0; w < worldState.size(); ++w )
        {
            // delta function
            if( worldState[w] == k )
            {
                // add train data to mean
                for( int i = 0; i < worldState.size(); ++i )
                {
                    mean[k][i] += trainData[w][i];
                }
            }
        }
    }
    for( int k = 0; k < mean.size(); ++k )
    {
        for( int j = 0; j < mean[0].size(); ++j )
            mean[k][j] = mean[k][j] / stat[k];
    }

    // set variance
    vector<vector<vector<double>>> var( num_class, vector<vector<double>>(
        num_variable, vector<double>(num_variable,0)));
    for( int k = 1; k < num_class; ++k )
    {
        for( int w = 0; w < worldState.size(); ++w )
        {
            if( worldState[w] == k )
            {
                // calculate x_i - mean_k
                vector<double> diff( num_variable );
                for( int i = 0; i < diff.size(); ++i )
                    diff[i] = trainData[w][i] - mean[k][i];
                // fill the covariance matrix
                for( int i = 0; i < diff.size(); ++i )
                {
                    for( int j = 0; j < diff.size(); ++j )
                    {
                        var[k-1][j][i] += diff[i]*diff[j];
                    }
                }
            }
        }
    }
    for( int k = 0; k < num_class; ++k )
    {
        for( int i = 0; i < num_variable; ++i )
        {
            for( int j = 0; j < num_variable; ++j )
            {
                var[k][j][i] = var[k][j][i] / stat[k];
            }
        }
    }

    // set prior
    vector<double> lambda( num_class );
    for( int k = 0; k < num_class; ++k )
        lambda[k] = stat[k] / trainData.size();

    // compute likelihood for new point
    vector<double> l( num_class );
    for( int k = 0; k < num_class; ++k )
    {
        // compute determinant
        double det; 
        // adding diagonals
        for( int j = 0; j < var[k][0].size(); ++j )
        {
            double temp = 1;
            int i, m = j;
            for( int n = 0; n < var[k][0].size(); ++n )
            {
                temp *= var[k][m][i];
                i++, m++;
                if( i == var[k][0].size() )
                    i = 0;
                if( m == var[k][0].size() )
                    m = 0;
            }
            det += temp;
        }
        // subtracting diagonals

    }
    vector<double> result;
    return result;
}

int main()
{
    vector< vector<double> > data( 2, vector<double>(3,0) );
    test( data );
    return 0;
}

