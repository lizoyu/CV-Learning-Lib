/*************************************************************************
	> File Name: 4.4.cpp - maximum likelihood learning of categorical
	> Author: Chungyuen Li
	> Mail: lizoyu1@gmail.com
	> Created Time: 2017年01月20日 星期五 09时11分25秒
 ************************************************************************/

#include<iostream>
#include<cmath>
#include<vector>
using namespace std;
using std::vector;

vector<double> MLforCat ( vector<double>::iterator begin,
                         vector<double>::iterator end, int num_cat)
{
    vector<double> stat( num_cat );
    double size;
    for( vector<double>::iterator iter = begin; iter != end; ++iter )
    {    
        stat[*iter - 1]++;
        size++;
    }

    for( vector<double>::iterator iter = stat.begin();
        iter != stat.end(); ++iter )
        *iter = *iter / size;

    return stat;
}

int main()
{
    vector<double> testData;
    for( int i = 0; i < 2; ++i )
        testData.push_back(1);
    for( int i = 0; i < 5; ++i )
        testData.push_back(2);
    for( int i = 0; i < 3; ++i )
        testData.push_back(3);
    vector<double> result = MLforCat(testData.begin(), testData.end(), 3);
    for(vector<double>::iterator iter = result.begin();
        iter != result.end(); ++iter )
        cout << *iter << " ";
    cout << endl;
    return 0;
}
