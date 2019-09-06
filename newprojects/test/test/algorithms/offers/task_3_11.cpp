//
// Created by WithHeart on 2019-02-13.
//

#include "task_3_11.h"

#include <iostream>
#include <vector>
#include <stack>
#include <queue>
using namespace std;

void swap(int &a,int &b){
    int temp = a;
    a = b;
    b = temp;
}

bool duplicate(int numbers[],int len,int *duplication){
    if(numbers == nullptr || len<=0){
        return false;
    }
    for(int i=0;i<len;++i){
        if(numbers[i]<0 || numbers[i] > len-1){
            return false;
        }
    }
    for(int i=0;i<len;++i){
        while(i != numbers[i]){
            if(numbers[i] == numbers[numbers[i]]){
                *duplication = numbers[i];
                return true;
            }
            swap(numbers[i],numbers[numbers[i]]);

        }
    }
    return false;
}


bool Find(int target, vector<vector<int> > array) {
    int rows = array.size();
    if(rows<=0){
        return false;
    }
    int columns = array[0].size();
    if(columns<=0){
        return false;
    }
    int i=columns-1,j=0;
    while(i>=0 && j<rows){
        if(array[j][i] > target){
            i--;
        }else if(array[j][i]<target){
            ++j;
        }else{
            return true;
        }
    }
    return false;
}

void replaceSpace(char *str,int length) {
    if(str == nullptr || length<=0){
        return;
    }
    int space_count = 0;
    for(int i=0;i<length;++i){
        if(str[i] == ' '){
            space_count ++;
        }
    }
    if(space_count==0){
        return;
    }
    int new_length = length + 2*space_count;
    int j = new_length-1;
    int i = length-1;
    while(i>=0 && j>i){
        if(str[i] == ' '){
            str[j--] = '0';
            str[j--] = '2';
            str[j--] = '%';
        }else{
            str[j--] = str[i];
        }
        i--;
    }
}


struct ListNode{
    int val;
    ListNode *next;
    ListNode(int val):val(val),next(NULL){}
};

vector<int> printListFromTailToHead(ListNode* head) {
    vector<int>outputs;
    if(head == nullptr){
        return outputs;
    }
    stack<ListNode *> inputs;
    while(head!= nullptr){
        inputs.push(head);
        head = head->next;
    }
    while(!inputs.empty()){
        outputs.push_back(inputs.top()->val);
        inputs.pop();
    }
    return outputs;
}
struct TreeNode{
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};


//重点关注这一题：不一定非要按照书上的代码写
TreeNode* constructTreeCode(const vector<int>&pre, int pre_start,int pre_end,
                            const vector<int>&vin, int vin_start,int vin_end){
    if(pre_start > pre_end || vin_start > vin_end){
        return nullptr;
    }
    TreeNode *root = new TreeNode(pre[pre_start]);
    root->left = root->right = nullptr;

    for(int vin_root = vin_start;vin_root <= vin_end;++vin_root){
        if(vin[vin_root] == pre[pre_start]){
            //构建左子树
            root->left = constructTreeCode(pre,pre_start + 1,pre_start + vin_root - vin_start,vin,vin_start,vin_root-1);
            //构建右子树
            root->right = constructTreeCode(pre,pre_start + vin_root - vin_start + 1,pre_end,vin,vin_root + 1,vin_end);

        }
    }
    return root;

}

TreeNode* reConstructBinaryTree(vector<int> pre,vector<int> vin) {
    int pre_len = pre.size();
    int vin_len = vin.size();
    return constructTreeCode(pre,0,pre_len-1,vin,0,vin_len-1);
}


class Solution
{
public:
    void push(int node) {
        stack1.push(node);
    }

    int pop() {
        if(stack2.empty()){
            while(!stack1.empty()){
                stack2.push(stack1.top());
                stack1.pop();
            }
        }
        int ret = stack2.top();
        stack2.pop();
        return ret;

    }

private:
    stack<int> stack1;
    stack<int> stack2;
};


int minNumberInRotateArray(vector<int> rotateArray) {
    int len = rotateArray.size();
    if(len==0){
        return 0;
    }
    int min = INT_MAX;
    for(int i=0;i<len;++i){
        if(rotateArray[i] < min){
            min = rotateArray[i];
        }
    }
    return min;
}

int Fibonacci(int n) {
    if(n <= 1){
        return n;
    }
    int first = 0;
    int second = 1;
    int result = 0;
    for(int i=2;i<=n;++i){
        result = first + second;
        first = second;
        second = result;
    }
    return result;
}

int jumpFloor(int number) {
    if(number <= 2){
        return number;
    }
    int first = 0;
    int second = 1;
    int result = 0;
    for(int i=3;i<=number;++i){
        result = first + second;
        first = second;
        second = result;
    }
    return result;
}

int jumpFloorII(int number) {
    if(number<=2){
        return number;
    }
    vector<int>numbers(number+1,0);
    numbers[0] = 0;
    numbers[1] = 1;
    numbers[2] = 2;
    for(int i = 3;i<=number;++i){
        int cur_sum = 0;
        for(int j=0;j<i;++j){
            cur_sum += numbers[j];
        }
        numbers[i] = cur_sum + 1;
    }
    return numbers[number];
}

int  NumberOf1(int n) {
    int flag =1;
    int count = 0;
    while(flag){ //备注：这里是考虑flag是否是0，而不是大于0
        if(n & flag){
            count ++;
        }
        flag<<=1;
    }
    return count;
}

double Power(double base, int exponent) {
    if(base == 0){
        return base;
    }
    if(exponent == 0){
        return 1;
    }
    int cur = exponent > 0 ? exponent : -exponent;
    double result = 1;
    while(cur--){
        result *= base;
    }
    if(exponent < 0){
        result = 1.0/result;
    }
    return result;

}

//void swap(int &a,int &b){
//    int temp = a;
//    a = b;
//    b = temp;
//}

void reOrderArray(vector<int> &array) {
    queue<int>temp;
    int index = 0;
    int len = array.size();
    for(int i=0;i<len;++i){
        if(array[i]%2){
            swap(array[index],array[i]);
            index++;
        }else{
            temp.push(array[i]);
        }
    }
    while(!temp.empty()){
        array[index++] = temp.front();
        temp.pop();
    }
}


ListNode* FindKthToTail(ListNode* pListHead, unsigned int k) {
    ListNode *first = pListHead;
    ListNode *last = pListHead;
    if(pListHead == nullptr || k<=0){
        return nullptr;
    }
    int num = 1;
    while(num <= k && first != nullptr){
        num ++;
        first = first->next;
    }
    if(num <= k && first == nullptr){ //k比联表长，异常情况，不需要处理
        return nullptr;
    }
    if(num - k == 1 && first== nullptr){ // 删除头结点的情况
        return pListHead;
    }
    while(first->next!= nullptr){
        first = first->next;
        last = last->next;
    }
    last = last->next;
    return last;
}

ListNode* ReverseList(ListNode* pHead) {
    if(pHead == nullptr || pHead->next == nullptr){
        return pHead;
    }
    ListNode *cur = pHead;
    ListNode *pre = nullptr;
    ListNode *newHead = nullptr;
    while(cur != nullptr){
        ListNode *next = cur->next;
        if(next == nullptr){
            newHead = cur;
        }
        cur->next = pre;
        pre = cur;
        cur = next;
    }
    return newHead;
//    stack<ListNode*> temp;
//    ListNode *newHead = pHead;
//    while(newHead->next){
//        temp.push(newHead);
//        newHead = newHead->next;
//    }
//    ListNode *tempNode = newHead;
//    while(!temp.empty()){
//        ListNode *cur = temp.top();
//        temp.pop();
//        tempNode->next = cur;
//        tempNode = cur;
//    }
//    tempNode->next= nullptr; //注意这里一定要置为null，要不然会出现循环的现象...
//    return newHead;
}

ListNode* Merge(ListNode* pHead1, ListNode* pHead2)
{
    if(!pHead1){
        return pHead2;
    }
    if(!pHead2){
        return pHead1;
    }
    ListNode *newHead = nullptr;
    if(pHead1->val <= pHead2->val){
        newHead = pHead1;
        newHead->next = Merge(pHead1->next,pHead2);
    }else{
        newHead = pHead2;
        newHead->next = Merge(pHead1,pHead2->next);
    }
    return newHead;
}

bool isSubTree(TreeNode* pRoot1,TreeNode* pRoot2){
    if(!pRoot2){
        return true;
    }
    if(!pRoot1){
        return false;
    }
    if(pRoot1->val == pRoot2->val){
        return isSubTree(pRoot1->left,pRoot2->left) && isSubTree(pRoot1->right,pRoot2->right);
    }else{
        return false;
    }
}

bool HasSubtree(TreeNode* pRoot1, TreeNode* pRoot2)
{
    if(!pRoot1 || !pRoot2){
        return false;
    }
    bool result = false;
    if(pRoot1->val == pRoot2->val){
        result = isSubTree(pRoot1,pRoot2);
        if(!result){
            result = HasSubtree(pRoot1->left,pRoot2);
        }
        if(!result){
            result = HasSubtree(pRoot1->right,pRoot2);
        }
    }
    return result;
}

void Mirror(TreeNode *pRoot) {
    if(!pRoot || (!pRoot->left && !pRoot->right)){
        return;
    }
    TreeNode *temp = pRoot->left;
    pRoot->left = pRoot->right;
    pRoot->right = temp;
    if(pRoot->left){
        Mirror(pRoot->left);
    }
    if(pRoot->right){
        Mirror(pRoot->right);
    }
}

vector<int> printMatrix(vector<vector<int> > matrix) {
    //充分考虑特殊情况：
    // 1. 只有一行
    // 2. 只有一列
    // 3. 行大于列
    // 4. 列大于行
    vector<int>result;
    int len = matrix.size();
    if(len == 0){
        return result;
    }
    if(matrix[0].size() == 1){
        for(int i=0;i<len;++i){
            result.push_back(matrix[i][0]);
        }
        return result;
    }
    if(len == 1){
        return matrix[0];
    }
    int rowUpper = 0,rowDown = matrix.size()-1;
    int colLeft = 0,colRight = matrix[0].size() - 1;
    int i = 0,j = 0;
    while(true){
        if(colLeft == colRight){ //偶数
            result.push_back(matrix[rowUpper][colLeft]);
            break;
        }
        if(colLeft > colRight|| rowDown < rowUpper){ //奇数
            break;
        }
        i = rowUpper;
        j = colLeft;
        while(j <= colRight){
            result.push_back(matrix[i][j]);
            j ++ ;
        }
        rowUpper ++;

        i = rowUpper;
        j = colRight;
        if( rowDown < rowUpper){
            break;
        }
        while(i <= rowDown){
            result.push_back(matrix[i][j]);
            i++;
        }
        colRight --;

        i = rowDown;
        j = colRight;
        if( colLeft > colRight){
            break;
        }
        while(j >= colLeft && colLeft <= colRight){
            result.push_back(matrix[i][j]);
            j--;
        }
        rowDown --;

        i = rowDown;
        j = colLeft;
        if( rowDown < rowUpper){
            break;
        }
        while(i >= rowUpper){
            result.push_back(matrix[i][j]);
            i--;
        }
        colLeft ++ ;
    }
    return result;
}

bool IsPopOrder(vector<int> pushV,vector<int> popV) {
    stack<int>pushData;
    int pushLen = pushV.size();
    int popLen = popV.size();
    int curIndex = 0;
    for(int i=0;i<popLen;++i){
        if(curIndex >= popLen && pushData.empty()) {
            return false;
        }
        if(!(!pushData.empty() && popV[i] == pushData.top())){
            while(curIndex < popLen && pushV[curIndex] != popV[i]){
                pushData.push(pushV[curIndex++]);
            }
            if(curIndex >= popLen){
                return false;
            }
            if(pushV[curIndex] == popV[i]){
                curIndex ++;
            }
        }else{
            pushData.pop();
        }
    }
    return true;
}


vector<int> PrintFromTopToBottom(TreeNode* root) {
    queue<TreeNode *>nodes;
    vector<int> res;
    if(!root){
        return res;
    }
    nodes.push(root);
    while(!nodes.empty()){
        TreeNode* cur = nodes.front();
        res.push_back(cur->val);
        if(cur->left){
            nodes.push(cur->left);
        }
        if(cur->right){
            nodes.push(cur->right);
        }
        nodes.pop();
    }
    return res;
}

bool verifySeq(const vector<int> &seq,int start,int end){
    if(start >= end){
        return true;
    }
    int root = seq[end];
    int pivot = start;
    for(;pivot < end;++pivot){
        if(seq[pivot] > root){
            break;
        }
    }
    bool res = true;
    for(int i= pivot;i < end;++i){
        if(seq[i] < root){
            return false;
        }
    }

    if(pivot > start){
        res = verifySeq(seq,start,pivot-1);
    }

    if(pivot < end){
        res = verifySeq(seq,pivot,end-1);
    }
    return res;
}

bool VerifySquenceOfBST(vector<int> sequence) {
    if(sequence.size() == 0){
        return false;
    }
    return verifySeq(sequence,0,sequence.size()-1);
}

void FindPath(TreeNode* root,int expectNumber,int crtNumber,vector<vector<int> > &res,vector<int> &curNode){
    crtNumber += root->val;
    curNode.push_back(root->val);
    if(!root->left && !root->right && expectNumber == crtNumber){
        //find a way
        res.push_back(curNode);
    }else if(root->left || root->right){
        if(root->left){
            FindPath(root->left,expectNumber,crtNumber,res,curNode);
        }
        if(root->right){
            FindPath(root->right,expectNumber,crtNumber,res,curNode);
        }
    }
    curNode.pop_back(); //注意退栈的位置
}

vector<vector<int> > FindPath(TreeNode* root,int expectNumber) {
    vector<vector<int> > res;
    vector<int> curNode;
    if(!root){
        return res;
    }
    FindPath(root,expectNumber,0,res,curNode);
    return res;
}



//int main(){
//    ListNode *one = new ListNode(8);
//    ListNode *two = new ListNode(11);
//    ListNode *three = new ListNode(7);
//    one->next = two;
//    two->next = three;
//
//    ListNode *one1 = new ListNode(8);
//    ListNode *two1 = new ListNode(11);
//    ListNode *three1 = new ListNode(7);
//    one1->next = two1;
//    two1->next = three1;
//    ListNode *head = Merge(one,one1);
////    while (head!= nullptr){
////        cout<<head->val<<endl;
////        head = head->next;
////    }
//
//    vector<int> row{1,2,3,4};
//    vector<int> row1{5,3,9,10};
////    vector<int> row2{9};
////    vector<int> row3{13};
//    vector<vector<int>> inputs;
//    inputs.push_back(row);
//    inputs.push_back(row1);
////    inputs.push_back(row2);
////    inputs.push_back(row3);
//    vector<int > res = printMatrix(inputs);
//    for(vector<int >::iterator it = res.begin();it != res.end();it ++){
//        cout<<*it<<endl;
//    }
//}