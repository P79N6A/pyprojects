//
//  1.cpp
//  test
//
//  Created by WithHeart on 2018/11/3.
//  Copyright © 2018年 WithHeart. All rights reserved.
//

#include <stdio.h>

class Point{
public:
    Point(float x= 0.0):_x(x){}
    float x(){
        return _x;
    }
    void x(float xval){
        _x = xval;
    }
protected:
    float _x;
};


class Point2d:public Point{
public:
    Point2d(float x=0.0,float y = 0.0):Point(x), _y(y){}
    float y(){
        return _y;
    }
    void y(float yval){_y = yval;}
protected:
    float _y;
};
