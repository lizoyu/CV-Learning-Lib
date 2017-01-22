/*************************************************************************
	> File Name: 4.5.cpp - MAP learning of categorical
	> Author: Chungyuen Li
	> Mail: lizoyu1@gmail.com
	> Created Time: 2017年01月22日 星期日 08时57分58秒
 ************************************************************************/

#include<iostream>
#include<vector>
using namespace std;
using std::vector;

vector<double> MAPforCat ( vector<double>::iterator begin,
                          vector<double>::iterator end,
                          vector<double> alpha)
{
    vector<double> stat(alpha.size());
    double size_data;
    for( vector<double>::iterator iter = begin;
       iter != end; ++iter)
    {
        stat[*iter-1]++;
        size_data++;
    }
    
    vector<double> lambda(alpha.size());
    double sum_alpha;
    for( vector<double>::iterator iter = alpha.begin();
        iter != alpha.end(); ++iter )
        sum_alpha += *iter;
    for( int i = 0; i < alpha.size(); ++i )
        lambda[i] = ( stat[i] -1 + alpha[i] ) / ( size_data - alpha.size() + sum_alpha );
    
    return lambda;
}

int main()
{
    vector<double> trainData;
    for( int i = 0; i < 3; ++i )
        trainData.push_back(1);
    for( int i = 0; i < 2; ++i )
        trainData.push_back(2);
    for( int i = 0; i < 5; ++i )
        trainData.push_back(3);
    vector<double> alpha(3,4);
    
    vector<double> result = MAPforCat( trainData.begin(),
                                     trainData.end(), alpha);

    cout << result[0] << " " << result[1] << " " << result[2] << endl;
    return 0;
}
