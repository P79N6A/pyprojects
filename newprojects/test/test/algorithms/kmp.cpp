//
//  kmp.cpp
//  test
//
//  Created by WithHeart on 2018/11/6.
//  Copyright © 2018年 WithHeart. All rights reserved.
//

#include <stdio.h>
#include <iostream>
using namespace std;

int get_kmp(string s,string p,int *next){
    int i= 0,j=0;
    int len_s = s.length();
    int len_p = s.length();
    while(i<len_s && i<len_p){
        if(s[i] == p[j]){
            ++i;
            ++j;
        }else{
            j = next[j];
        }
    }
    if (i>=len_s){
        return j-i;
    }else{
        return -1;
    }
}

int* get_next(string p){
    int len_p = p.length();
    int next[len_p];
    int i = 0,j=-1;
    if(i<len_p){
        if(next[i] == next[j]){
            ++ i;
            ++ j;
            next[i] = j;
        }else{
            j = next[j];
        }
    }
    
    return next;
}
