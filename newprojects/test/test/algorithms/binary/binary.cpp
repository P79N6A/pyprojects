//
// Created by WithHeart on 2019-02-24.
//

#include "binary.h"
#include <vector>
#include <iostream>

using namespace std;

// 原始的二分查找
int binary(const vector<int>&arr,int key){
    int left  = 0;
    int right = arr.size() - 1;
    while(left <= right){
        int mid = left + ((right - left) >> 1);
        if(arr[mid] > key){
            right = mid + 1;
        }else if(arr[mid] < key){
            left = mid - 1;
        }else{
            return mid;
        }
    }
    return -1;
}

//--------------------------------------
//二分查找总结：
//1. 首先确定是返回left 还是返回 right
//跳出循环时，left是大于right的
//所以如何判断时返回left的值还是right的值呢？
//方法：返回第一个是返回left的值，返回最后一个是返回right的值
//2. 第二个是需要判断符号以及right还是left值的修改
//方法：
// 如果是返回left，那么符号就是针对right时的更新，也就是题目要求的符号情况下更新right的值为right = arr[mid] -1 ，然后else分支修改left只能；
// 如果返回是right，那么符号针对的就是left时的更新；
//--------------------------------------


// 第一个与key相等的元素
int binaryFirst(const vector<int>&arr,int key){
    int left = 0;
    int right = arr.size()-1;
    while(left <= right){
        int mid = left + ((right - left) >> 1);
        if(arr[mid] >= key){
            right = mid-1;
        }else{
            left = mid + 1;
        }
    }
    if(left < arr.size() && arr[left] == key){
        return left;
    }else{
        return -1;
    };
}

//最后一个与key相等的元素
int binaryLast(const vector<int> &arr,int key){
    int left = 0;
    int right = arr.size()-1;
    while(left<= right){
        int mid = left + ((right - left) >> 1);
        if(arr[mid]<=key){
            left = mid +1;
        }else{
            right = mid-1;
        }
    }
    if(right>=0 && arr[right]==key){
        return right;
    }else{
        return -1;
    }
}


//第一个大于或者等于key的元素
int binaryFirstNoLessThan(const vector<int> &arr,int key){
    int left = 0;
    int right = arr.size()-1;
    while(left <= right){
        int mid = left + ((right - left) >> 1);
        if(arr[mid] >= key){
            right = mid - 1;
        }else{
            left = mid + 1;
        }
    }
    return left;
}

//第一个大于key的元素
int binaryFirstMoreThan(const vector<int> & arr,int key){
    int left = 0;
    int right = arr.size();
    while(left <= right){
        int mid = left + ((right - left) >> 1);
        if(arr[mid] > key){
            right = mid - 1;
        }else{
            left = mid + 1;
        }
    }
    return left;
}

//最后一个小于等于key的元素
int binaryLastNoMoreThan(const vector<int> &arr,int key){
    int left = 0;
    int right = arr.size() - 1;
    while(left <= right){
        int mid = left + ((right - left) >> 1);
        if(arr[mid] <= key){
            left = mid + 1;
        }else{
            right = mid - 1;
        }
    }
    return right;
}

//最后一个小于key的元素
int binaryLastLessThan(const vector<int> &arr,int key){
    int left = 0;
    int right = arr.size() - 1;
    while(left <= right){
        int mid = left + ((right - left) >> 1);
        if(arr[mid] < key){
            left = mid + 1;
        }else{
            right = mid - 1;
        }
    }
    return right;
}


int main(){
    vector<int>arr = {1,1,2,3,4,5,6,6,6,6,7};
    cout <<"binary:"<< binary(arr,6) << endl;
    cout <<"binaryFirst:"<< binaryFirst(arr,6) << endl;
    cout <<"binaryFirstMoreThan:"<< binaryFirstMoreThan(arr,6) << endl;
    cout <<"binaryFirstNoLessThan:"<< binaryFirstNoLessThan(arr,6) << endl;

    cout <<"binaryLast:"<< binaryLast(arr,6) << endl;
    cout <<"binaryLastLessThan:"<< binaryLastLessThan(arr,6) << endl;
    cout <<"binaryLastNoMoreThan:"<< binaryLastNoMoreThan(arr,6) << endl;
}
