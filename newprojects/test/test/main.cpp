//
//  main.cpp
//  test
//
//  Created by WithHeart on 2018/10/27.
//  Copyright © 2018年 WithHeart. All rights reserved.
//

#include <iostream>
#include <list>
#include <map>
#include <set>
#include <stack>

using namespace std;

list<string> getSimiUrls(string url){
    list<string>result ;
    return result;
}

int getUrlType(string url){
    return 1;
}

set<string>getAllSimiUrl(string url){
    set<string>urls;
    stack<string> inUrls;
    while(inUrls.size() != 0){
        string curUrl = inUrls.top();
        list<string> curSimUrls = getSimiUrls(curUrl);
        
        inUrls.pop();
    }
    return urls;
}

map<int,string> getALlSimUrlTypes(string url){
    map<int,string> result;
    return result;
}


//int main(int argc, const char * argv[]) {
//    // insert code here...
//    std::cout << "Hello, World!\n";
//    return 0;
//}
