#include <iostream>
#include <stack>
#include <string>
#include <vector>
#include <algorithm>
#include <queue>
#include <unordered_set>

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

void sort_impl(std::vector<int> &arr, int left, int right){
    if(left < right) {
        int p = partation(arr, left, right);
        sort_impl(arr, left, p - 1);
        sort_impl(arr, p + 1, right);
    }  
}

void quick_sort(std::vector<int> &arr) {
    if (arr.empty()) {
        return;
    }
    
    sort_impl(arr, 0, int(arr.size()) - 1);
}

int remove_duplicate(int* arr, int len) {
    if (arr == nullptr || len == 0) {
        return -1;
    }
    
    std::cout << len << std::endl;
    int valid = 0;
    for (int i = valid + 1; i < len; ++i) {
        if (arr[i] != arr[valid]) {
            arr[++valid] = arr[i];
            std::cout << valid << std::endl;
            
        } 
    }
    return valid;
}

std::string up(std::string s, const int& idx) {
    if (s[idx] == '9') {
        s[idx] = '0';
    } else {
        s[idx] += 1;
    }
    return std::string(s);
}

std::string down(std::string s, const int& idx) {
    if (s[idx] == '0') {
        s[idx] = '9';
    } else {
        s[idx] -= 1;
    }
    return std::string(s);
}

int open_lock(const std::vector<std::string>& deadends, const std::string& target) {
    int step = 0;
    std::unordered_set<std::string> deadends_set(deadends.begin(), deadends.end());
    std::unordered_set<std::string> record;
    std::queue<std::string> q;
    
    q.push("0000");
    record.insert("0000");
    while(!q.empty()) {
        std::queue<std::string> p(q);
        int size = q.size();
        for (int i = 0; i < size; ++i) {
            std::string cur = q.front();
            q.pop();

            if (deadends_set.count(cur)) {
                continue;
            }

            if (cur == target) {
                return step;
            }
            
            for (int j = 0; j < 4; ++j) {
                std::string change = up(cur, j);
                if (record.count(change)) {
                    continue;
                }
                q.push(change);
                record.insert(change);
                change = down(cur, j);
                if (record.count(change)) {
                    continue;
                }
                q.push(change);
                record.insert(change);
            }
        }
        step++;
        if (p.size() == q.size()) {
            std::cout <<  (p == q) << std::endl;
        }
    }
    return step;
}

int find_max_common_str(const std::string& first, const std::string& second) {
    int m = int(first.size());
    int n = int(second.size());
    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1));
    for (int i = 1; i <= m; ++i) {
        char first_c = first[i - 1];
        for (int j = 1; j <= n; ++j){
            char second_c = second[j - 1];
            if (first_c == second_c) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = std::max(dp[i -1][j], dp[i][j - 1]);
            }
        }
    }
    return dp[m][n];
}

int find_max_common_sub_str(const std::string& first, const std::string& second) {
    int m = int(first.size());
    int n = int(second.size());
    int max_len = 0;
    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1));
    for (int i = 1; i <= m; ++i) {
        char first_c = first[i - 1];
        for (int j = 1; j <= n; ++j){
            char second_c = second[j - 1];
            if (first_c == second_c) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
                /*if (dp[i - 1][j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                    
                } else {
                    dp[i][j] = 1;
                }*/
                max_len = std::max(max_len, dp[i][j]);
            } 
        }
    }
    return max_len;
}

std::string longest_huiwen(const std::string& arr) {
    int size = arr.size();
    int start = 0;
    int max_len = 0;
    if (size < 2) {
        return arr;
    }
    std::vector<std::vector<int>> dp(size, std::vector<int>(size));
    for (int i = 0; i < size; ++i) {
        dp[i][i] = 1;
    }

    for (int win = 2; win <= size; ++win) {
        for (int left = 0; left < size; ++left) {
            int right = left + win - 1;
            if (right >= size) {
                break;
            }
            if (arr[left] != arr[right]) {
                dp[left][right] = 0;
            } else {
                if (win <= 3) {
                    dp[left][right] = 1;
                } else {
                    dp[left][right] = dp[left + 1][right-1];
                }
            }
            if (dp[left][right] && win > max_len) {
                max_len = win;
                start = left;
            }
        }
    }
    return arr.substr(start, max_len);
}

std::string longest_huiwen_twice(const std::string& arr) {
    int size = arr.size();
    int start = 0;
    int max_len = 0;
    if (size < 2) {
        return arr;
    }
    std::vector<std::vector<int>> dp(size, std::vector<int>(size));

    for (int right = 0; right < size; ++right) {
        for (int left = 0; left <= right; ++left) {
            int win = right - left + 1;
            if (win == 1) {
                dp[left][right] = 1;
            } else if (win == 2) {
                dp[left][right] = (arr[left] == arr[right]); 
            } else {
                dp[left][right] = dp[left + 1][right-1] && (arr[left] == arr[right]);
            }
            if (dp[left][right] && win > max_len) {
                max_len = win;
                start = left;
            }
        }
    }
    return arr.substr(start, max_len);
}

int center_longest(const std::string& arr, 
                                   int left,
                                   int right) {
    while(left >= 0 && right < arr.size() && arr[left] == arr[right]) {
        left--;
        right++;
    }
    // 终止是left多向左移动一步，right多向右移动一步
    return right -1  - (left + 1) + 1;
}

std::string longest_huiwen_new(const std::string& arr) {
    int size = arr.size();

    if (size < 2) {
        return arr;
    }
    int start = 0, end = 0;
    for (int i = 0; i < size; ++i) {

        int len1 = center_longest(arr, i, i);
        int len2 = center_longest(arr, i, i + 1);
        int len = std::max(len1, len2);
        if (len > (end - start)) {
            start = i - (len - 1)/ 2;  // 为兼容长度为1或2扩散
            end = i + len / 2;
        }
    }
    

    return arr.substr(start, end - start + 1);
}


int vol_water(const std::vector<int>& arr) {
    int n =  arr.size();
    int sum = 0;
    for (int i = 1; i < n -1; ++i) {
        int left_max = 0;
        for (int j = 0; j <= i; ++j) {
            if (arr[j] > left_max) {
                left_max = arr[j];
            }
        }
        int right_max = 0;
        for (int j = i; j < n; ++j) {
            if (arr[j] > right_max) {
                right_max = arr[j];
            }
        }
        sum += (std::min(left_max, right_max) - arr[i]);
    }
    return sum;
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

    /*std::vector<int> arr = {0, 3, 5, 7, 4, 2, 1, 6};
    for (int i = 0; i < arr.size(); ++i) {
        std::cout << i + 1 << ":" << find_k_max(arr, i + 1) << std::endl;
    }*/
    /*int arr[] = {1, 2, 3, 3, 4, 4, 5};
    int len = sizeof(arr) / sizeof(int);
    int valid = remove_duplicate(arr, len);*/
    /*std::vector<int> arr = {0, 3, 5, 7, 4, 2, 1, 6};
    quick_sort(arr);
    for (int i = 0; i < arr.size(); ++i) {
        std::cout << arr[i] << "," << std::endl;
    }*/
    /*std::vector<std::string> deadends = {"0201", "0101", "0102", "1212", "2002"};
    std::string target = "0202";
    std::cout << "need step:" << open_lock(deadends, target) << std::endl;*/

    /*std::string first = "abcde";
    std::string second   = "bcd";
    int value = find_max_common_str(first, second);
    std::cout << find_max_common_sub_str(first, second);*/
    //std::string arr = "babad";
    std::string arr = "adbbdcmabbamc";
    std::cout << longest_huiwen_twice(arr) << std::endl;
    /*std::vector<int> arr = {0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1};
    std::cout << vol_water(arr) << std::endl;*/
    return 0;
}
