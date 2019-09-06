//
//  fib.cpp
//  test
//
//  Created by WithHeart on 2018/11/15.
//  Copyright © 2018年 WithHeart. All rights reserved.
//

#include <iostream>
#include <vector>
using namespace std;


//递归的解法：自顶向下
int fib1(int n){
    if(n==0 || n==2){
        return n;
    }else{
        return fib1(n-1) + fib1(n-2);
    }
}

//非递归的解法：自底向上 --> 备忘录
int fib2(int n){
    vector<int> data(100);
    data[0] = 0;
    data[1] = 1;
    for(int i = 2;i <= n;++i){
        data[i] = data[i-1] + data[i-2];
    }
    return data[n];
}

//压缩方式的备忘录
int fib3(int n){
    int temp1 = 0;
    int temp2 = 1;
    int value = 0;
    for(int i=2;i<=n;++i){
        value = temp1 + temp2;
        temp1 = temp2;
        temp2 = value;
    }
    return value;
}
