#include <iostream>
#include <stack>
#include <string>
#include <vector>
#include <algorithm>

bool is_valid(const std::string& s) {
    if (s.empty()) {
        return true;
    }
    std::stack<int> st;
    for (int i = 0; i < s.size(); ++i) {
        if (s[i] == '(') {
            st.push(s[i]);
        } else if (s[i] == ')' && !st.empty()) {
            st.pop();
        } else {
            return false;
        }
    }
    if (st.empty()) {
        return true;
    } else {
        return false;
    }
}

void back_track(const std::vector<int>& arr,
                std::vector<int>& track, 
                std::vector<int>& flag,
                std::vector<std::vector<int> >& res) {
    if (track.size() == arr.size()) {
        res.push_back(track);
        return;
    }
    for (int i = 0; i < arr.size(); ++i) {
        if (!flag[i]) {
            flag[i] = 1;
            track.push_back(arr[i]);
            back_track(arr, track, flag, res);
            track.pop_back();
            flag[i] = 0;
        }
    }
}

std::vector<std::vector<int> > permutation(const std::vector<int>& arr) {
    std::vector<std::vector<int> > res;
    if (arr.empty()) {
        return res;
    }
    std::vector<int> track;
    std::vector<int> flag(arr.size(), 0);
    back_track(arr, track, flag, res);
    return res;
}

int partation(std::vector<int> &arr, int left, int right){
    int middle = left + (right - left) / 2;
    int cur = arr[middle];
    std::swap(arr[middle], arr[right]);
    while (left < right) {
        while(left < right && arr[left] <= cur) {
            left++;
        }
        std::swap(arr[left], arr[right]);
        while(left < right && arr[right] >= cur) {
            right--;
        }
        std::swap(arr[left], arr[right]);
    }
    arr[left] = cur;
    return left;
}
int quick_select(std::vector<int> &arr, int left, int right, int k){
    int p = partation(arr, left, right);
    if (p == k) {
        return arr[k];
    } else if (p > k) {
        return quick_select(arr, left, p - 1, k);
    } else {
        return quick_select(arr, p + 1, right, k);
    }

}

int find_k_max(std::vector<int> &arr, int k) {
    if (arr.empty()) {
        return -1;
    }
    
    return quick_select(arr, 0, int(arr.size()) - 1, int(arr.size()) - k);
}

int main(int argc, char* argv[]) {
    //std::cout << "hello word" << std::endl;
    //printf("hello word!");
    //std::string s = ")(()))";
    //std::cout << is_valid(s) << std::endl;
    
    /*std::vector<int> arr = {1, 2, 3};
    std::vector<std::vector<int> > res = permutation(arr);
    for (int i = 0; i < res.size(); ++i) {
        std::cout << "(";
        for (int j = 0; j < res[i].size(); ++j) {
            std::cout << res[i][j];
        }
        std::cout << ")" << std::endl;
    }*/

    std::vector<int> arr = {0, 3, 5, 7, 4, 2, 1, 6};
    for (int i = 0; i < arr.size(); ++i) {
        std::cout << i + 1 << ":" << find_k_max(arr, i + 1) << std::endl;
    }
    

    return 0;
}
