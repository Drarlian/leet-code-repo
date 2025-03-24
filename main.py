from typing import Optional, List
from numpy import median


def romanToInt(s: str) -> int:
    options = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    other_options = {'IV': 4, 'IX': 9, 'XL': 40, 'XC': 90, 'CD': 400, 'CM': 900}

    result = 0
    indice = 0
    while True:
        if indice == len(s):
            return result

        if indice + 1 < len(s):
            if other_options.get(s[indice] + s[indice + 1], False):
                result += other_options[s[indice] + s[indice + 1]]
                indice += 2
                continue

        result += options[s[indice]]
        indice += 1

    return result


def removeElement(nums: List[int], val: int) -> int:
    contador = nums.count(val)

    while val in nums:
        nums.remove(val)

    for x in range(contador):
        nums.append(val)

    return len(nums) - contador


def merge(nums1: List[int], m: int, nums2: List[int], n: int) -> None:
    """
    Do not return anything, modify nums1 in-place instead.
    """
    maior = m if m > n else n
    for _ in range(maior):
        if len(nums1) == m and len(nums2) == n:
            break

        if len(nums1) != m:
            nums1.pop()

        if len(nums2) != n:
            nums2.pop()

    nums1.extend(nums2)
    nums1.sort()

    print(nums1)
    print(nums2)


# merge([0,0,0,0,0], 0, [1,2,3,4,5], 5)


def removeDuplicates(nums: List[int]) -> int:
    contador = 0

    while contador < len(nums) - 1:
        if nums[contador] == nums[contador + 1]:
            nums.remove(nums[contador])
        else:
            contador += 1

    print(nums)

    return len(nums)


# removeDuplicates([0,0,1,1,1,2,2,3,3,4])
# removeDuplicates([1,1,2])


def climbStairs(n: int) -> int:
    """
    A resposta para essa pergunta esta no Fibonacci!
    """

    infos = {'actual': 1, 'past': 0, 'temp': 0}

    for c in range(n):
        infos['temp'] = infos['actual']
        infos['actual'] = infos['actual'] + infos['past']
        infos['past'] = infos['temp']

    return infos['actual']


# print(climbStairs(2))
# print(climbStairs(3))


def maxArea(height: List[int]) -> int:
    ponti_ini = 0
    ponti_fini = len(height) - 1
    better_area = (ponti_fini - ponti_ini) * (height[ponti_fini] if height[ponti_ini] > height[ponti_fini] else height[ponti_ini])

    for i in range(len(height)):
        if (ponti_fini - ponti_ini) <= 0:
            break

        area_atual = (ponti_fini - ponti_ini) * (height[ponti_fini] if height[ponti_ini] > height[ponti_fini] else height[ponti_ini])

        if area_atual > better_area:
            better_area = area_atual

        if height[ponti_ini] > height[ponti_fini]:
            ponti_fini -= 1
        else:
            ponti_ini += 1

    return better_area


# maxArea([1,8,6,2,5,4,8,3,7])
# print(maxArea([1,1]))
# print(maxArea([8,7,2,1]))


def reverseString(s: List[str]) -> None:
    x = len(s) - 1

    for i in range(len(s)):
        if x <= i:
            break

        s[i], s[x] = s[x], s[i]
        x -= 1

    print(s)


# reverseString(["h","e","l","l","o"])      # -> ["o","l","l","e","h"]
# reverseString(["H","a","n","n","a","h"])  # -> ["h","a","n","n","a","H"]


def isPowerOfTwo(n: int) -> bool:
    while True:
        if n == 2 or n == 1:
            return True

        if n % 2 == 1 or n == 0:
            return False

        n = n / 2


# print(isPowerOfTwo(1))   # -> True
# print(isPowerOfTwo(16))  # -> True
# print(isPowerOfTwo(3))   # -> False
# print(isPowerOfTwo(8))   # -> True
# print(isPowerOfTwo(32))   # -> True


def moveZeroes(nums: List[int]) -> None:
    """
    Do not return anything, modify nums in-place instead.
    """
    i = 0
    for j in range(len(nums)):
        if nums[j] != 0:
            nums[i], nums[j] = nums[j], nums[i]
            i += 1


# moveZeroes([0,1,0,3,12])  # -> [1,3,12,0,0]
# moveZeroes([0])           # -> [0]
# moveZeroes([0,0,1])       # -> [1,0,0]
# moveZeroes([1,0,1])       # -> [1,1,0]


def reverseVowels(s: str) -> str:
    if len(s) == 1:
        return s

    vowles = ['a', 'e', 'i', 'o', 'u']
    s = [letter for letter in s]

    ini = 0
    fini = len(s) - 1

    while ini <= fini:
        if s[ini].lower() in vowles and s[fini].lower() in vowles:
            s[ini], s[fini] = s[fini], s[ini]

            ini += 1
            fini -= 1

        elif s[ini].lower() in vowles and s[fini].lower() not in vowles:
            fini -= 1

        elif s[fini].lower() in vowles and s[ini].lower() not in vowles:
            ini += 1
        else:
            ini += 1
            fini -= 1

    return ''.join(s)


# print(reverseVowels("IceCreAm"))  # -> "AceCreIm"
# print(reverseVowels("leetcode"))  # -> "leotcede"
# print(reverseVowels("ai"))  # -> "ia"
# print(reverseVowels("!!!"))  # -> "ia"
# print(reverseVowels(" apG0i4maAs::sA0m4i0Gp0"))  # -> " ipG0A4mAas::si0m4a0Gp0"


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


# print(detectCapitalUse("USA"))   # -> True
# print(detectCapitalUse("FlaG"))  # -> False


def countSegments(s: str) -> int:
    if len(s.strip()) == 0:
        return 0

    ls = s.strip().split(' ')
    result = []
    for c in ls:
        if c != '':
            result.append(c)

    return len(result)


# print(countSegments("Hello, my name is John"))   # -> 5
# print(countSegments("Hello"))                    # -> 1
# print(countSegments(", , , ,        a, eaefa"))  # -> 6


def sortPeople(names: List[str], heights: List[int]) -> List[str]:
    union = list(zip(names, heights))
    union.sort(key=lambda x: x[1], reverse=True)
    result = [x[0] for x in union]
    return result


# print(sortPeople(names=["Mary","John","Emma"], heights=[180,165,170]))  # -> ["Mary","Emma","John"]
# print(sortPeople(names=["Alice","Bob","Bob"], heights=[155,185,150]))    # -> ["Bob","Alice","Bob"]


def intersectII(nums1: List[int], nums2: List[int]) -> List[int]:
    result = []

    cont1 = 0
    cont2 = 0

    nums1.sort()
    nums2.sort()
    while True:
        if cont1 == len(nums1) or cont2 == len(nums2):
            break

        if nums1[cont1] == nums2[cont2]:
            result.append(nums1[cont1])
            cont1 += 1
            cont2 += 1
        elif nums1[cont1] > nums2[cont2]:
            cont2 += 1
        else:
            cont1 += 1

    return result


# print(intersectII(nums1=[1,2,2,1], nums2=[2,2]))      # -> [2,2]
# print(intersectII(nums1=[4,9,5], nums2=[9,4,9,8,4]))  # -> [4,9] or [9,4]
# print(intersectII(nums1=[1,2], nums2=[1,1]))          # -> [1]
# print(intersectII(nums1=[2,1], nums2=[1,2]))          # -> [1,2]
# print(intersectII(nums1=[3,1,2], nums2=[1,1]))        # -> [1]


def findDifference(nums1: List[int], nums2: List[int]) -> List[List[int]]:
    hash_num1 = {value for value in nums1}
    hash_num2 = {value for value in nums2}

    for num in hash_num1.copy():
        if num in hash_num2:
            hash_num1.remove(num)
            hash_num2.remove(num)

    return [list(hash_num1), list(hash_num2)]


# print(findDifference(nums1 = [1,2,3], nums2 = [2,4,6]))  # -> [[1,3],[4,6]]
# print(findDifference(nums1 = [1,2,3,3], nums2 = [1,1,2,2]))  # -> [[3],[]]


def findWords(words: List[str]) -> List[str]:
    first = 'eiopqrtuwy'
    second = 'adfghjkls'
    third = 'bcmnvxz'

    teste = (1, 2, 3, 4)
    teste.count(1)

    result = []
    for word in words:
        temp = set(word)
        temp_dict = {'first': 0, 'second': 0, 'third': 0}

        for letter in temp:
            if list(temp_dict.values()).count(0) != 3 and list(temp_dict.values()).count(0) != 2:
                break

            if letter.lower() in first:
                temp_dict['first'] += 1

            elif letter.lower() in second:
                temp_dict['second'] += 1

            elif letter.lower() in third:
                temp_dict['third'] += 1

        if temp_dict['first'] == len(temp):
            result.append(word)

        if temp_dict['second'] == len(temp):
            result.append(word)

        if temp_dict['third'] == len(temp):
            result.append(word)

    return result


"""
l1="qwertyuiop"
l2="asdfghjkl"
l3="zxcvbnm"
res=[]
for word in words:
    w=word.lower()
    if len(set(l1+w))==len(l1) or len(set(l2+w))==len(l2) or len(set(l3+w))==len(l3) :
        res.append(word)
return res
"""
print(findWords(["Hello","Alaska","Dad","Peace"]))  # -> ["Alaska","Dad"]
print(findWords(["omk"]))                           # -> []
print(findWords(["adsdf","sfd"]))                   # -> ["adsdf","sfd"]
