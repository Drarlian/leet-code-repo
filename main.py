from typing import List


def strStr(haystack: str, needle: str) -> int:
    return haystack.find(needle)


# print(strStr('sadbutsad', 'sad'))
# print(strStr('leetcode', 'leeto'))


def plusOne(digits: List[int]) -> List[int]:
    n = ''.join([str(x) for x in digits])
    return [int(x) for x in str(int(n) + 1)]


# print(plusOne([1,2,3]))    # -> [1,2,4]
# print(plusOne([4,3,2,1]))  # -> [4,3,2,2]
# print(plusOne([9]))        # -> [1,0]
# print(plusOne([9,9]))      # -> [1,0,0]


def addBinary(a: str, b: str) -> str:
    return bin(int(a, 2) + int(b, 2))[2:]


# print(addBinary('11', '1'))  # -> 100
# print(addBinary('1010', '1011'))  # -> 10101


def singleNumber(nums: List[int]) -> int:
    lista_removidos = set()

    for item in nums:
        if item not in lista_removidos:
            if nums.count(item) == 1:
                return item
            else:
                lista_removidos.add(item)


# print(singleNumber([2,2,1]))      # -> 1
# print(singleNumber([4,1,2,1,2]))  # -> 4
# print(singleNumber([1]))          # -> 1


# a = 40
# b = 40
#
# print(a ^ b)
#
# if a ^ b == 0:  # Se XOR der 0, os números são iguais
#     print("Os números são iguais")


def containsDuplicate(nums: List[int]) -> bool:
    contain = set()

    for num in nums:
        if num in contain:
            return True

        contain.add(num)

    return False


# print(containsDuplicate([1,2,3,1]))  # -> True
# print(containsDuplicate([1,2,3,4]))  # -> False
# print(containsDuplicate([1,1,1,3,3,4,3,2,4,2]))  # -> True


def intersection(nums1: List[int], nums2: List[int]) -> List[int]:
    # return list(set(nums1).intersection(set(nums2)))

    mp = {}
    for num in nums1:
        mp[num] = mp.get(num, 0) + 1

    result = []
    for num in nums2:
        if num in mp:
            result.append(num)
            del mp[num]

    return result


# print(intersection([1,2,2,1], [2,2]))      # -> [2]
# print(intersection([4,9,5], [9,4,9,8,4]))  # -> [9,4]
# print(intersection([3,1,2], [1]))          # -> [1]
# print(intersection([1,2], [1,1]))          # -> [1]


def findTheDifference(s: str, t: str) -> str:
    s = sorted(s)
    t = sorted(t)

    for indice, letra in enumerate(s):
        if letra != t[indice]:
            return t[indice]

    return t[-1]


# print(findTheDifference('abcd', 'abcde'))  # -> e
# print(findTheDifference('', 'y'))  # -> y
# print(findTheDifference('a', 'aa'))  # -> a


def addDigits(num: int) -> int:
    if len(str(num)) == 1:
        return num

    while True:
        num = str(sum([int(num) for num in str(num)]))

        if len(num) == 1:
            return int(num)


# print(addDigits(38))  # -> 2
# print(addDigits(0))   # -> 0


def validPalindrome(s: str) -> bool:
    if s == s[::-1] or len(s) == 0 or len(s) == 1 or len(s) == 2:
        return True

    temp = [c for c in s]

    for c in range(len(s)):
        temp.pop(c)

        if ''.join(temp) == ''.join(reversed(temp)):
            return True

        temp.insert(c, s[c])

    return False


# print(validPalindrome("aba"))   # -> true
# print(validPalindrome("abca"))  # -> true  (You could delete the character 'c'.)
# print(validPalindrome("abc"))   # -> false


def repeatedSubstringPattern(s: str) -> bool:
    if len(s) == 0 or len(s) == 1:
        return True

    for caracter in range(len(s) - 1):
        if s.replace(s[:caracter+1], '') == '':
            return True

    return False


# print(repeatedSubstringPattern('abab'))          # -> True
# print(repeatedSubstringPattern('aba'))           # -> False
# print(repeatedSubstringPattern('abcabcabcabc'))  # -> True


def thirdMax(nums: List[int]) -> int:
    nums = sorted(list(set(nums)), reverse=True)

    if len(nums) < 3:
        return max(nums)

    return nums[2]


# print(thirdMax([3,2,1]))    # -> 1
# print(thirdMax([1,2]))      # -> 2
# print(thirdMax([2,2,3,1]))  # -> 1


def findNonMinOrMax(nums: List[int]) -> int:
    if len(nums) <= 2:
        return -1

    return sorted(nums)[1]


# print(findNonMinOrMax([3,2,1,4]))  # -> 2
# print(findNonMinOrMax([1,2]))      # -> -1
# print(findNonMinOrMax([2,1,3]))    # -> 2


def detectCapitalUse(word: str) -> bool:
    if word.lower() == word or word.upper() == word or word.capitalize() == word:
        return True

    return False


print(detectCapitalUse("USA"))   # -> True
print(detectCapitalUse("FlaG"))  # -> False
