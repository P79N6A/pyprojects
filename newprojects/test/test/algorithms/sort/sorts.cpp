////
//// Created by WithHeart on 2019-02-12.
////
//
//#include "sorts.h"
//#include <iostream>
//#include <vector>
//#include "math.h"
//using namespace std;
//
//void swap(int &a,int &b){
//    int temp = a;
//    a = b;
//    b = temp;
//}
//
///*
//（无序区，有序区）。从无序区通过交换找出最大元素放到有序区前端。
//选择排序思路：
//1. 比较相邻的元素。如果第一个比第二个大，就交换他们两个。
//2. 对每一对相邻元素作同样的工作，从开始第一对到结尾的最后一对。这步做完后，最后的元素会是最大的数。
//3. 针对所有的元素重复以上的步骤，除了最后一个。
//4. 持续每次对越来越少的元素重复上面的步骤，直到没有任何一对数字需要比较。
//*/
//void bubble(vector<int> &inputs){
//    int len = inputs.size();
//    for(int i = 0;i<len-1;++i){
//        for(int j=0;j<len-i-1;++j){
//            if(inputs[j] > inputs[j+1]){
//                swap(inputs[j],inputs[j+1]);
//            }
//        }
//    }
//}
//
///*
//（有序区，无序区）。在无序区里找一个最小的元素跟在有序区的后面。对数组：比较得多，换得少。
//选择排序思路：
//1. 在未排序序列中找到最小（大）元素，存放到排序序列的起始位置
//2. 从剩余未排序元素中继续寻找最小（大）元素，然后放到已排序序列的末尾
//3. 以此类推，直到所有元素均排序完毕
//*/
//void select(vector<int>&inputs){
//    int len = inputs.size();
//    int min_index;
//    for(int i=0;i<len-1;++i){
//        min_index = i;
//        for(int j=i+1;j<len;++j){
//            if(inputs[j] < inputs[min_index]){
//                min_index = j;
//            }
//        }
//        swap(inputs[i],inputs[min_index]);
//    }
//}
//
//
///*
//（有序区，无序区）。把无序区的第一个元素插入到有序区的合适的位置。对数组：比较得少，换得多。
//插入排序思路：
//1. 从第一个元素开始，该元素可以认为已经被排序
//2. 取出下一个元素，在已经排序的元素序列中从后向前扫描
//3. 如果该元素（已排序）大于新元素，将该元素移到下一位置
//4. 重复步骤3，直到找到已排序的元素小于或者等于新元素的位置
//5. 将新元素插入到该位置后
//6. 重复步骤2~5
//*/
//void insert(vector<int>&inputs){
//    int len = inputs.size();
//    for(int i=1;i<len;++i){
//        int des_index = i-1;
//        int cur = inputs[i];
//        while(des_index>=0 && inputs[des_index]>cur){
//            inputs[des_index + 1] = inputs[des_index];
//            des_index --;
//        }
//        inputs[des_index+1] = cur; //注意：这里是des_index+1，不是des_index
//    }
//}
//
//// 希尔排序：每一轮按照事先决定的间隔进行插入排序，间隔会依次缩小，最后一次一定要是1。
//void shell(vector<int>&inputs){
//    int len = inputs.size();
//    int gap = 1;
//    while(gap < len/3){
//        gap = gap*3 + 1;
//    }
//    while(gap >= 1){
//        for(int i=gap;i<len;++i){
//            int des_index = i-gap;
//            int cur = inputs[i];
//            while(des_index>=0 && inputs[des_index]>cur){
//                inputs[des_index + gap] = inputs[des_index];
//                des_index -= gap;
//            }
//            inputs[des_index + gap] = cur;
//        }
//        gap/=3;
//    }
//}
//
//
//// 归并排序：把数据分为两段，从两段中逐个选最小的元素移入新数据段的末尾。可从上到下或从下到上进行。
//void merge_recursive(vector<int> &inputs,vector<int>&target,int start,int end){
//    if(start>=end){
//        return;
//    }
//    int len = end-start;
//    int mid = (len>>1) + start;
//    int start1 = start,end1 = mid;
//    int start2 = mid+1,end2= end;
//    merge_recursive(inputs,target,start1,end1);
//    merge_recursive(inputs,target,start2,end2);
//    int k = start;
//    while(start1<=end1 && start2<=end2){
//        target[k++] = inputs[start1] > inputs[start2]?inputs[start2++]:inputs[start1++];
//    }
//    while(start1<=end1){
//        target[k++] = inputs[start1++];
//    }
//    while(start2<=end2){
//        target[k++] = inputs[start2++];
//    }
//    for(int i = start;i<=end;++i){
//        inputs[i] = target[i];
//    }
//}
//
//void merge(vector<int> &inputs){
//    int len = inputs.size();
//    vector<int>target(len); //临时存储，为了和其他排序算法保持一致，返回结果还在放在inputs里面了
//    merge_recursive(inputs,target,0,len-1);
//}
//
//
///*
//（小数，基准元素，大数）。在区间中随机挑选一个元素作基准，将小于基准的元素放在基准之前，大于基准的元素放在基准之后，再分别对小数区与大数区进行排序。
//快速排序思路：
//1. 选取第一个数为基准
//2. 将比基准小的数交换到前面，比基准大的数交换到后面
//3. 对左右区间重复第二步，直到各区间只有一个数
//*/
//void quick_recursive(vector<int>&inputs,int low,int high){
//    if(low>=high){
//        return;
//    }
//    int first = low;
//    int last = high;
//    int key = inputs[first];
//    while(first < last){
//        while(first<last && inputs[last]>=key){
//            last--;
//        }
//        if(first<last){
//            inputs[first++] = inputs[last];
//        }
//        while(first<last && inputs[first]<=key){
//            first++;
//        }
//        if(first<last){
//            inputs[last--]=inputs[first];
//        }
//    }
//    inputs[first]=key; //上述循环退出的条件是first == last
//    quick_recursive(inputs,low,first);
//    quick_recursive(inputs,first+1,low);
//}
//
//void quick(vector<int>&inputs){
//    int len = inputs.size();
//    quick_recursive(inputs,0,len-1);
//}
//
//
//// 堆排序：（最大堆，有序区）。从堆顶把根卸出来放在有序区之前，再恢复堆。
//void heap_max(vector<int>&inputs,int start,int end){
//    int dad = start;
//    int son = dad*2+1;
//    while(son<=end){
//        if(son+1<=end && inputs[son] < inputs[son+1]){
//            son ++;
//        }
//        if(inputs[dad] >= inputs[son]){
//            return;
//        }else{
//            swap(inputs[dad],inputs[son]);
//            //继续子节点和孙子节点的比较
//            dad = son;
//            son = dad*2 + 1;
//        }
//    }
//}
//
//void heap(vector<int>&inputs){
//    int len = inputs.size();
//    //初始化，i從最後一個父節點開始調整
//    for(int i=len/2-1;i>=0;--i){
//        heap_max(inputs,i,len-1);
//    }
//    //经过上面的初始化之后，vector的第一个元素已经是最大的元素，所以将其移动到最后，然后调整其他元素
//    //先將第一個元素和已经排好的元素前一位做交換，再從新調整(刚调整的元素之前的元素)，直到排序完畢
//    for(int i=len-1;i>=0;--i){
//        swap(inputs[0],inputs[i]);
//        heap_max(inputs,0,i-1);
//    }
//}
//
///*****************
//计数排序：统计小于等于该元素值的元素的个数i，于是该元素就放在目标数组的索引i位（i≥0）。
//计数排序基于一个假设，待排序数列的所有数均为整数，且出现在（0，k）的区间之内。
//如果 k（待排数组的最大值） 过大则会引起较大的空间复杂度，一般是用来排序 0 到 100 之间的数字的最好的算法，但是它不适合按字母顺序排序人名。
//计数排序不是比较排序，排序的速度快于任何比较排序算法。
//时间复杂度为 O（n+k），空间复杂度为 O（n+k）
//算法的步骤如下：
//1. 找出待排序的数组中最大和最小的元素
//2. 统计数组中每个值为 i 的元素出现的次数，存入数组 C 的第 i 项
//3. 对所有的计数累加（从 C 中的第一个元素开始，每一项和前一项相加）
//4. 反向填充目标数组：将每个元素 i 放在新数组的第 C[i] 项，每放一个元素就将 C[i] 减去 1
//*****************/
//void count(vector<int>&inputs){
//    int len = inputs.size();
//    if(len<=0){
//        return;
//    }
//    vector<int>target(len,0);
//    int max = inputs[0];
//    for(int i=0;i<len;++i){
//        if(inputs[i]>max){
//            max = inputs[i];
//        }
//    }
//    vector<int>counts(max+1,0);
//    for(int i=0;i<len;++i){
//        counts[inputs[i]] ++;
//    }
//    for(int i=1;i<=max;++i){
//        counts[i]+=counts[i-1];
//    }
//    for(int i=len-1;i>=0;--i){
//        target[--counts[inputs[i]]] = inputs[i];
//    }
//    for(int i=0;i<len;++i){
//        inputs[i] = target[i];
//    }
//}
//
//
////todo:桶排序和基数排序就先不写了 ~
//
//int main(){
//    vector<int> inputs{19,9,2,7,7,3,11};
//    int len = inputs.size();
//    cout<<"before sort: ";
////    for(vector<int>::iterator it = inputs.begin();it!= inputs.end();++it){
////        cout<<*it<<endl;
////    }
//    for(int i=0;i<len;++i){
//        cout<<inputs[i]<<" ";
//    }
//    cout<<endl;
//    count(inputs);
//    cout<<"after sort: ";
//    for(int i=0;i<len;++i){
//        cout<<inputs[i]<<" ";
//    }
//    cout<<endl;
//}
//
//
