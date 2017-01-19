/*************************************************************************
	> File Name: 4.1.cpp - Maximum likelihood learning of normal distribution
	> Author: Chungyuen Li
	> Mail: lizoyu1@gmail.com
	> Created Time: 2017年01月18日 星期三 06时06分32秒
 ************************************************************************/

#include<iostream>
#include<vector>
#include<cmath>
using namespace std;
using std::vector;

vector<double> MLforNormal ( vector<double>::iterator begin,
                            vector<double>::iterator end)
{
    double mean, var, sum, sum_deviSquare, size;

    // calculate the mean of the training data
    for (vector<double>::iterator iter = begin;
         iter != end; ++iter)
    {
        sum += *iter;
        size++;
    }
    mean = sum / size;

    // calculate the variance using the mean above
    for (vector<double>::iterator iter = begin;
        iter != end; ++iter)
        sum_deviSquare += pow(*iter - mean, 2);
    var = sum_deviSquare / size;
    vector<double> result(2);

    // save the mean and variance in vector 'result'
    result[0] = mean;
    result[1] = var;
    return result;
}

// testing
int main ()
{
    vector<double> test;
    for(int i = 1; i < 11; ++i)
        test.push_back(i);
    vector<double> result(2);
    result = MLforNormal(test.begin(),test.end());
    cout << result[0] << "," << result[1] << endl;
    return 0;
}
