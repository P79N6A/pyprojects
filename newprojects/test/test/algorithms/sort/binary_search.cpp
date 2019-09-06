//
// Created by WithHeart on 2019-02-12.
//

#include "binary_search.h"
#include<iostream>
#include<vector>
using namespace std;

int binary_search(const vector<int> &inputs,int target){
    int len = inputs.size();
    int low = 0;
    int high = len-1;
    int mid = 0;
    while(low<=high){
        mid = low + (high - low)/2;
        if(inputs[mid]<target){
            low = mid +1;
        }else if(inputs[mid] > target){
            high = mid-1;
        }else{
            return mid;
        }
    }
    return -1;
}
