# class Solution:
#     def convert(self, s, numRows):
#         """
#         :type s: str
#         :type numRows: int
#         :rtype: str
#         """
#         if numRows == 1:
#             return s
#         _list = []
#         res = ''
#         unit = 2 * numRows - 2
#         a = len(s) // unit
#         for i in range(a):
#             _list.append(self.helper(s[i * unit:(i + 1) * unit], numRows))
#         _list.append(self.helper(s[a * unit:], numRows))
#         print(_list)
#         for i in range(numRows):
#             for L in _list:
#                 for l in L:
#                     res += l[i]
#         print(res)
#         return res
#
#
#     def helper(self, s, n):
#         temp = [[''] * n for _ in range(n-1)]
#         length = len(s)
#         for i in range(length):
#             if i < n:
#                 temp[0][i] = s[i]
#             else:
#                 temp[i - n + 1][2 * n - i - 2] = s[i]
#         return temp


class Solution:
    def convert(self, s, numRows):
        """
        :type s: str
        :type numRows: int
        :rtype: str
        """
        if numRows == 1:
            return s
        _list = [''] * numRows
        index = 0
        step = 1
        for i in s:
            _list[index] += i
            if index == 0:
                step = 1
            elif index == numRows - 1:
                step = -1
            index += step
        return ''.join(_list)


class Solution:
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        # for i in range(len(nums)):
        #     for j in range(i+1,len(nums)):
        #         if nums[i] + nums[j] == target:
        #             return i,j
        # return None
        nums = sorted([(nums[i], i) for i in range(len(nums))])
        print(nums)
        left = 0
        right = len(nums) - 1
        while left < right:
            t = nums[left][0] + nums[right][0]
            if t == target:
                return nums[left][1], nums[right][1]
            elif t < target:
                left += 1
            else:
                right -= 1
        return -1, -1


class Solution:
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """

        _dict = {}
        start = 0
        max_len = 0
        for i, j in enumerate(s):
            if j in _dict and _dict[j] >= start:
                max_len = max(max_len, i - start)
                start = _dict[j] + 1
            _dict[j] = i
        max_len = max(max_len, len(s) - start)#可能最后的最长字符串不走if条件
        return max_len


class Solution:
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        string = ""
        for i in range(len(s)):
            res = self.__isPalindrome(s, i, i)
            if len(res) > len(string):
                string = res
            if i < len(s) - 1:
                res = self.__isPalindrome(s, i, i + 1)
                if len(res) > len(string):
                    string = res
        return string

    def __isPalindrome(self, s, i, j):
        while i >= 0 and j <= len(s) - 1 and s[i] == s[j]:
            i -= 1
            j += 1
        return s[i + 1:j]

class Solution:
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        # if x > 0:
        #     flag = 1
        # else:
        #     x = -x
        #     flag = -1
        # _list = []
        # while x > 0:
        #     _list.append(x % 10)
        #     x = x // 10
        # y = 0
        # while len(_list) > 0:
        #     y = y * 10
        #     y += _list.pop(0)
        # y = y * flag
        # if y >= -2 ** 31 and y <= 2 ** 31 - 1:
        #     return y
        # else:
        #     return 0

        y = (-1,1)[x > 0] * int(str(abs(x))[::-1])
        if y >= -2 ** 31 and y <= 2 ** 31 - 1:
            return y
        else:
            return 0


class Solution:
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        _min = 0
        _max = len(height) - 1
        max_area = (_max - _min) * min(height[_min], height[_max])
        start = _min
        stop = _max
        while _min < _max:
            if height[_min] > height[_max]:
                area = (_max - _min) * height[_max]
                # max_area = max(area,max_area)
                if area > max_area:
                    start = _min
                    stop = _max
                    max_area = area
                _max -= 1
            else:
                area = (_max - _min) * height[_min]
                # max_area = max(area,max_area)
                if area > max_area:
                    start = _min
                    stop = _max
                    max_area = area
                _min += 1
        print(start, stop)
        return max_area

class Solution:
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if len(strs) == 0:
            return ''
        strs = sorted(strs)
        res = ''
        for i in strs[0]:
            if strs[-1].startswith(res+i):
                res += i
            else:
                break
        return res


class Solution:
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """

        if digits == "":
            return digits

        num_key = {"2": "abc", "3": "def", "4": "ghi", "5": "jkl", "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz"}
        self._out = []
        _list = [num_key[s] for s in digits]
        print(_list)
        self.helper(_list, '')
        return self._out

    def helper(self, _list, chosed):
        if len(_list) == 0:
            self._out.append(chosed)
            return
        for l in _list[0]:
            self.helper(_list[1:], chosed + l)

if __name__ == "__main__":
    s = Solution()
    # print(s.convert("PAYPALISHIRING", 3))
    # print(s.maxArea([1, 8, 6, 2, 5, 4, 8, 3, 7]))
    a = '23'
    # a.remove(3)
    print(s.letterCombinations(a))
    print(a)
    print(a[3:])
