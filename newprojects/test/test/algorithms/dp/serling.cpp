//
//  serling.cpp
//  test
//
//  Created by WithHeart on 2018/11/15.
//  Copyright © 2018年 WithHeart. All rights reserved.
//

#include <stdio.h>
#include <iostream>
#include <vector>
using namespace std;


// |--------------------------------------------|
// |    长度i   |  1 |  2 |  3 |  4  |  5  |  6  |
// |--------------------------------------------|
// |  价格p(i)  |  1 |  5 |  8 |  9  |  10 |  17 |
// |--------------------------------------------|
// 求解一种切割方案，使得收益最大
// 分析：针对每一个长度处，可以考虑是否切分（切分之后左边保持不变，右边执行递归操作），从左到右一次判断
// 推广题型：其他方式获取最大值：比如两端的乘积
//递归法
int serling1(int n,int p[]){
    if(n==0){
        return 0;
    }else{
        int max_value = INT_MAX;
        for(int i=1;i<=n;++i){
            max_value = max(max_value,p[i-1] + serling1(n-i,p));
        }
        return max_value;
    }
}

//非递归法 --> 备忘录法
int serling2(int n ,int p[],vector<int>interest){
    if(n==0){
        return 0;
    }else if(interest[n] > 0){
        return interest[n];
    }else{
        int max_value = INT_MAX;
        for(int i=1;i<=n;++i){
            max_value = max(max_value, p[i-1] + serling2(n-i, p,interest));
        }
        interest[n] = max_value;
        return max_value;
    }
}

int serling2_main(int n,int p[]){
    vector<int> interest(n,-1);
    return serling2(n, p,interest);
}


//非递归法 --> 自底向上
// [i: 1--> n]-----------i-------------
// [j:1 --> i]-------j---|
// 所以在计算interest[i]时所有的interest[i-j]都是已知的
int serling3(int n,int p[]){
    if(n == 0){
        return 0;
    }else{
        vector<int>interest(n+1,0);
        for (int i = 1;i<=n;++i){
            int max_value = INT_MAX;
            for(int j = 1;j<=i;++j){
                max_value = max(max_value, p[j-1] + interest[i-j]);
            }
            interest[i] = max_value;
        }
        return interest[n];
    }
}

