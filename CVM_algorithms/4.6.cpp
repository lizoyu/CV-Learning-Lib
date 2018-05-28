/*************************************************************************
	> File Name: 4.6.cpp - Bayesian learning of categorical
	> Author: Chungyuen Li
	> Mail: lizoyu1@gmail.com
	> Created Time: 2017年01月23日 星期一 08时57分44秒
 ************************************************************************/

#include<iostream>
#include<vector>
using namespace std;
using std::vector;

vector<double> BayeforCat ( vector<double>::iterator begin,
                           vector<double>::iterator end,
                           vector<double> alpha)
{
    vector<double> stat( alpha.size() );
    for( vector<double>::iterator iter = begin; iter != end; ++iter )
        stat[*iter - 1]++;

    vector<double> alpha_1;
    int i = 0;
    double sum_alpha;
    for( vector<double>::iterator iter = alpha.begin();
       iter != alpha.end(); ++iter )
    {    
        alpha_1.push_back( *iter + stat[i] );
        sum_alpha += alpha_1[i];
        i++;
    }
    
    vector<double> P;
    for( vector<double>::iterator iter = alpha_1.begin();
       iter != alpha_1.end(); ++iter )
        P.push_back( *iter / sum_alpha );

    return P;
}

int main()
{
    vector<double> testData;
    for( int i = 0; i < 3; ++i )
        testData.push_back(1);
    for( int i = 0; i < 2; ++i )
        testData.push_back(2);
    for( int i = 0; i < 5; ++i )
        testData.push_back(3);

    vector<double> alpha(3,4);
    vector<double> result = BayeforCat( testData.begin(), testData.end(),
                                      alpha );

    cout << result[0] << " " << result[1] << " " << result[2] << endl;
    return 0;
}
