from collections import defaultdict
from typing import Optional, List
from numpy import median

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


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
    better_area = (ponti_fini - ponti_ini) * (
        height[ponti_fini] if height[ponti_ini] > height[ponti_fini] else height[ponti_ini])

    for i in range(len(height)):
        if (ponti_fini - ponti_ini) <= 0:
            break

        area_atual = (ponti_fini - ponti_ini) * (
            height[ponti_fini] if height[ponti_ini] > height[ponti_fini] else height[ponti_ini])

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
        if s.replace(s[:caracter + 1], '') == '':
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


# print(findWords(["Hello","Alaska","Dad","Peace"]))  # -> ["Alaska","Dad"]
# print(findWords(["omk"]))                           # -> []
# print(findWords(["adsdf","sfd"]))                   # -> ["adsdf","sfd"]


def numJewelsInStones(jewels: str, stones: str) -> int:
    hash_response = dict()
    for stone in stones:
        hash_response[stone] = hash_response.get(stone, 0) + 1

    cont = 0
    for jewel in jewels:
        if jewel in hash_response:
            cont += hash_response[jewel]
            del hash_response[jewel]

    return cont


# print(numJewelsInStones(jewels="aA", stones="aAAbbbb"))  # -> 3
# print(numJewelsInStones(jewels="z", stones="ZZ"))        # -> 0


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def deleteDuplicates(head: Optional[ListNode]) -> Optional[ListNode]:
    # import copy
    # temp_head = copy.deepcopy(head)
    temp_head = head  # Atribuindo o temp_head para o endereço inicial de head.

    while head and head.next:
        if head.val == head.next.val:
            head.next = head.next.next
        else:
            head = head.next

    return temp_head


def selfDividingNumbers(left: int, right: int) -> List[int]:
    result = []
    for num in range(left, right + 1):
        if '0' in str(num):
            continue

        is_self_dividing = True
        for little_num in str(num):
            if num % int(little_num) != 0:
                is_self_dividing = False
                break

        if is_self_dividing:
            result.append(num)

    return result


# print(selfDividingNumbers(left=1, right=22))   # -> [1,2,3,4,5,6,7,8,9,11,12,15,22]
# print(selfDividingNumbers(left=47, right=85))  # -> [48,55,66,77]


def removeDuplicates(s: str) -> str:
    res = []
    for c in s:
        if res and res[-1] == c:
            res.pop()
        else:
            res.append(c)
    return "".join(res)


# print(removeDuplicates("abbaca"))  # -> "ca"
# print(removeDuplicates("azxxzy"))  # -> "ay"
# print(removeDuplicates("aaaaaaaa"))  # -> "# "


def dominantIndex(nums: List[int]) -> int:
    sorted_nums = sorted(nums, reverse=True)

    if sorted_nums[1] > (sorted_nums[0] / 2):
        return -1
    else:
        return nums.index(sorted_nums[0])


# print(dominantIndex([3,6,1,0]))  # -> 1
# print(dominantIndex([1,2,3,4]))  # -> -1


def flipAndInvertImage(image: List[List[int]]) -> List[List[int]]:
    result = [list(map(lambda x: 1 if x == 0 else 0, value[::-1])) for value in image]

    return result


# print(flipAndInvertImage([[1,1,0],[1,0,1],[0,0,0]]))                  # -> [[1,0,0],[0,1,0],[1,1,1]]
# print(flipAndInvertImage([[1,1,0,0],[1,0,0,1],[0,1,1,1],[1,0,1,0]]))  # -> [[1,1,0,0],[0,1,1,0],[0,0,0,1],[1,0,1,0]]


def transpose(matrix: List[List[int]]) -> List[List[int]]:
    result = [[] for _ in matrix[0]]

    for row in range(len(matrix[0])):
        for column in range(len(matrix)):
            result[row].append(matrix[column][row])

    return result


# print(transpose([[1,2,3],[4,5,6],[7,8,9]]))  # -> [[1,4,7],[2,5,8],[3,6,9]]
# print(transpose([[1,2,3],[4,5,6]]))          # -> [[1,4],[2,5],[3,6]]
# print(transpose([[1,2],[4,5],[7,8]]))        # -> [[1, 4, 7], [2, 5, 8]]


def reverseOnlyLetters(s: str) -> str:
    hash_map = {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u",
                "v", "w", "x", "y", "z"}

    array_s = [value for value in s]

    left = 0
    right = len(s) - 1

    while True:
        if left == right or (left == len(s) - 1) or right == 0 or left > right:
            break

        if array_s[left].lower() in hash_map and array_s[right].lower() in hash_map:
            array_s[left], array_s[right] = array_s[right], array_s[left]
            left += 1
            right -= 1
        elif array_s[left].lower() in hash_map and array_s[right].lower() not in hash_map:
            right -= 1
        elif array_s[left].lower() not in hash_map and array_s[right].lower() in hash_map:
            left += 1
        else:
            left += 1
            right -= 1

    return ''.join(array_s)


# print(reverseOnlyLetters("ab-cd"))                 # -> "dc-ba"
# print(reverseOnlyLetters("a-bC-dEf-ghIj"))         # -> "j-Ih-gfE-dcba"
# print(reverseOnlyLetters("Test1ng-Leet=code-Q!"))  # -> "Qedo1ct-eeLg=ntse-T


def isMonotonic(nums: List[int]) -> bool:
    if sorted(nums) == nums or sorted(nums, reverse=True) == nums:
        return True

    return False


# print(isMonotonic([1,2,2,3]))  # -> True
# print(isMonotonic([6,5,4,4]))  # -> True
# print(isMonotonic([1,3,2]))    # -> False


def reverseWords(s: str) -> str:
    temp_list = s.split(' ')

    for i in range(len(temp_list)):
        temp_list[i] = temp_list[i][::-1]

    return ' '.join(temp_list)


# print(reverseWords("Let's take LeetCode contest"))  # -> "s'teL ekat edoCteeL tsetnoc"
# print(reverseWords("Mr Ding"))                      # -> "rM gniD"


def commonChars(words: List[str]) -> List[str]:
    if len(words) == 1:
        return [letter for letter in words[0]]

    if len(words) < 1:
        return []

    words.sort()
    result = []

    for word in words[0]:
        exist_in_all = True
        for indice, palavras in enumerate(words[1:]):
            if word not in palavras:
                exist_in_all = False
                break
            else:
                words[indice + 1] = palavras.replace(word, '', 1)

        if exist_in_all:
            result.append(word)

    return result


# print(commonChars(["bella","label","roller"]))  # -> ["e","l","l"]
# print(commonChars(["cool","lock","cook"]))      # -> ["c","o"]
# print(commonChars(["cool"]))                    # -> ['c', 'o', 'o', 'l']


def isPalindrome(head: Optional[ListNode]) -> bool:
    tempList = []

    while head.next is not None:
        tempList.append(head.val)
        head = head.next

    tempList.append(head.val)
    return tempList == tempList[::-1]


# print(isPalindrome([1,2,2,1]))  # -> true
# print(isPalindrome([1,2]))      # -> false


def middleNode(head: Optional[ListNode]) -> Optional[ListNode]:
    tempData = []

    while head is not None:
        tempData.append(head)
        head = head.next

    return tempData[len(tempData) // 2]


# print(middleNode([1,2,3,4,5]))    # -> [3,4,5]
# print(middleNode([1,2,3,4,5,6]))  # -> [4,5,6]


def removeElements(head: Optional[ListNode], val: int) -> Optional[ListNode]:
    if head is None:
        return head

    init = None
    passNode = None

    while head is not None:
        if head.val != val:
            if passNode is None:
                passNode = ListNode(head.val)
                init = passNode
            else:
                passNode.next = ListNode(head.val)
                passNode = passNode.next

        head = head.next

    return init


# print(removeElements([1,2,6,3,4,5,6], 6))  # -> [1,2,3,4,5]
# print(removeElements([], 1))               # -> []
# print(removeElements([7,7,7,7], 7))        # -> []


def countOperations(num1: int, num2: int) -> int:
    operations = 0
    while (num1 > 0) and (num2 > 0):
        if num2 > num1:
            num2 -= num1
        else:
            num1 -= num2

        operations += 1

    return operations

# print(countOperations(num1 = 2, num2 = 3))    # -> 3
# print(countOperations(num1 = 10, num2 = 10))  # -> 1


def distinctIntegers(n: int) -> int:
    board = [n]
    for c in range(10):
        for item in range(board[-1], 1, -1):
            if board[-1] % item == 1:
                board.append(item)

    return len(set(board))


# print(distinctIntegers(5))  # -> 4
# print(distinctIntegers(3))  # -> 2


def arrayRankTransform(arr: List[int]) -> List[int]:
    hashResult = {}

    for indice, item in enumerate(sorted(set(arr))):
        hashResult[item] = indice + 1

    return [hashResult[item] for item in arr]


# print(arrayRankTransform([40,10,20,30]))                # -> [4,1,2,3]
# print(arrayRankTransform([100,100,100]))                # -> [1,1,1]
# print(arrayRankTransform([37,12,28,9,100,56,80,5,12]))  # -> [5,3,4,2,8,6,7,1,3]


def insertGreatestCommonDivisors(head: Optional[ListNode]) -> Optional[ListNode]:
    import math
    temp = head

    while head.next:
        g = math.gcd(head.val, head.next.val)

        head.next = ListNode(g, head.next)
        head = head.next.next

    return temp


# print(insertGreatestCommonDivisors([18,6,10,3]))  # -> [18,6,6,2,10,1,3]
# print(insertGreatestCommonDivisors([7]))          # -> [7]


def pivotArray(nums: List[int], pivot: int) -> List[int]:
    temp_less = []
    temp_bigger = []
    temp_equal = []

    for num in nums:
        if num < pivot:
            temp_less.append(num)
        elif num > pivot:
            temp_bigger.append(num)
        else:
            temp_equal.append(pivot)

    return temp_less + temp_equal + temp_bigger


# print(pivotArray([9,12,5,10,14,3,10], 10))  # -> [9,5,3,10,10,12,14]
# print(pivotArray([-3,4,3,2], 2))            # -> [-3,2,4,3]

def mergeNodes(head: Optional[ListNode]) -> Optional[ListNode]:
    new_node = None
    ini_new_node = None

    temp_value = 0
    find_zero = 0
    while head:
        if (find_zero == 1 and head.val == 0):
            if new_node is None:
                new_node = ListNode(temp_value)
                ini_new_node = new_node
            else:
                new_node.next = ListNode(temp_value)
                new_node = new_node.next

            find_zero = 1
            temp_value = 0

            head = head.next
            continue

        if head.val != 0:
            temp_value += head.val
        else:
            find_zero += 1

        head = head.next

    return ini_new_node


# print(mergeNodes([0,3,1,0,4,5,2,0]))  # -> [4,11]
# print(mergeNodes([0,1,0,3,0,2,2,0]))  # -> [1,3,4]

def mergeInBetween(list1: ListNode, a: int, b: int, list2: ListNode) -> ListNode:
    list1_ini = list1
    list2_ini = list2

    while list2.next:
        list2 = list2.next

    temp_cont = 0
    while list1:
        if temp_cont == a:
            temp = list1

            while temp:
                if temp_cont == b:
                    temp = temp.next
                    break
                temp_cont += 1
                temp = temp.next

            list1.val = list2_ini.val
            list1.next = list2_ini.next
            list2.next = temp
            break

        temp_cont += 1
        list1 = list1.next

    return list1_ini


# print(mergeInBetween([10,1,13,6,9,5], 3, 4, [1000000,1000001,1000002]))                  # -> [10,1,13,1000000,1000001,1000002,5]
# print(mergeInBetween([0,1,2,3,4,5,6], 2, 5, [1000000,1000001,1000002,1000003,1000004]))  # -> [0,1,1000000,1000001,1000002,1000003,1000004,6]

def reverseBetween(head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
    head_ini = head
    temp = []
    temp_head = None
    teste = True

    cont_posi = 1
    while head.next:
        if cont_posi == left or (cont_posi + 1) == left:
            if cont_posi == left:
                temp_head = head
                teste = False
            else:
                temp_head = head.next
                cont_posi += 1

            while temp_head:
                temp.append(temp_head.val)

                if cont_posi == right:
                    temp_head = temp_head.next
                    break

                cont_posi += 1
                temp_head = temp_head.next

            temp = temp[::-1]
            ini_new_part = None
            for i, v in enumerate(temp):
                if i == 0:
                    new_part = ListNode(temp[i])
                    ini_new_part = new_part
                else:
                    new_part.next = ListNode(temp[i])
                    new_part = new_part.next

            if not teste:
                head = ini_new_part
                head_ini = head
                continue
            else:
                head.next = ini_new_part

        cont_posi += 1
        head = head.next

    if temp_head:
        head.next = temp_head
    return head_ini


# print(reverseBetween([1,2,3,4,5], 2, 4))  # -> [1,4,3,2,5]
# print(reverseBetween([5], 1, 1))          # -> [5


def deleteDuplicates(head: Optional[ListNode]) -> Optional[ListNode]:
    result = []
    duplicates = []

    while head:
        if head.val in result:
            result.remove(head.val)
            duplicates.append(head.val)
        else:
            if head.val not in duplicates:
                result.append(head.val)
        head = head.next

    final_result = None
    ini_result = None
    for i, v in enumerate(result):
        if i == 0:
            final_result = ListNode(v)
            ini_result = final_result
        else:
            final_result.next = ListNode(v)
            final_result = final_result.next

    return ini_result


# print(deleteDuplicates([1,2,3,3,4,4,5]))  # -> [1,2,5]
# print(deleteDuplicates([1,1,1,2,3]))      # -> [2,3]


def modifiedList(nums: List[int], head: Optional[ListNode]) -> Optional[ListNode]:
    ini_head = head
    fast = head.next
    nums = set(nums)

    while head:
        if head.val in nums:
            if fast is not None:
                head.val = fast.val
                head.next = fast.next
            else:
                temp_ini = ini_head
                while temp_ini:
                    if temp_ini.next.val == head.val:
                        temp_ini.next = None
                        break
                    temp_ini = temp_ini.next
                break

        else:
            head = head.next

        if fast is not None:
            fast = fast.next

    return ini_head


# print(modifiedList([1,2,3], [1,2,3,4,5]))  # -> [4,5]
# print(modifiedList([1], [1,2,1,2,1,2]))    # ->  [2,2,2]
# print(modifiedList([5], [1,2,3,4]))        # -> [1,2,3,4]

def convertTemperature(celsius: float) -> List[float]:
    return [celsius + 273.15, celsius * 1.80 + 32]


# print(convertTemperature(36.50))   # -> [309.65000,97.70000]
# print(convertTemperature(122.11))  # -> [395.26000,251.79800]

def differenceOfSum(nums: List[int]) -> int:
    cont = 0

    for num in nums:
        if len(str(num)) == 0:
            cont += num
        else:
            cont += sum([int(c) for c in str(num)])

    return abs(sum(nums) - cont)


# print(differenceOfSum([1,15,6,3]))  # -> 9
# print(differenceOfSum([1,2,3,4]))   # -> 0

def firstPalindrome(words: List[str]) -> str:
    for word in words:
        if word == word[::-1]:
            return word

    return ""


# print(firstPalindrome(["abc","car","ada","racecar","cool"]))  # -> "ada"
# print(firstPalindrome(["notapalindrome","racecar"]))          # -> "racecar"
# print(firstPalindrome(["def","ghi"]))                         # -> ""


def removeOccurrences( s: str, part: str) -> str:
    while part in s:
        s = s.replace(part, '', 1)

    return s


# print(removeOccurrences('daabcbaabcbc', 'abc'))  # -> dab
# print(removeOccurrences('axxxxyyyyb', 'xy'))     # -> ab


def removeStars(s: str) -> str:
    stack = []

    for caractere in s:
        if caractere == '*':
            stack.pop()
            continue

        stack.append(caractere)

    return ''.join(stack)


# print(removeStars('leet**cod*e'))    # -> lecoe
# print(removeStars('erase*****'))     # -> ""


def checkValidString(s: str) -> bool:
    # *******

    lo = hi = 0
    for c in s:
        if c == '(':
            lo += 1
            hi += 1
        elif c == ')':
            lo -= 1
            hi -= 1
        else:
            lo -= 1
            hi += 1

        if hi < 0:
            return False

        lo = max(lo, 0)

    return lo == 0


# print(checkValidString("((((()(()()()*()(((((*)()*(**(())))))(())()())(((())())())))))))(((((())*)))()))(()((*()*(*)))(*)()"))  # -> True
# print(checkValidString("()"))    # -> True
# print(checkValidString("(*)"))   # -> True
# print(checkValidString("(*))"))  # -> True


def isValid(s: str) -> bool:
    stack = []

    for caractere in s:
        if caractere == ')' and len(stack) > 0:
            if stack[-1] == '(':
                stack.pop()
                continue
            else:
                return False

        if caractere == ']' and len(stack) > 0:
            if stack[-1] == '[':
                stack.pop()
                continue
            else:
                return False

        if caractere == '}' and len(stack) > 0:
            if stack[-1] == '{':
                stack.pop()
                continue
            else:
                return False

        stack.append(caractere)

    if len(stack) > 0:
        return False
    else:
        return True


# print(isValid('()'))      # -> True
# print(isValid('()[]{}'))  # -> True
# print(isValid('(]'))      # -> False
# print(isValid('([])'))    # -> True


def findNumbers(nums: List[int]) -> int:
    if len(nums) == 0:
        return 0

    result = 0

    for num in nums:
        if (len(str(num)) % 2) == 0:
            result += 1

    return result


# print(findNumbers([12,345,2,6,7896]))   # -> 2
# print(findNumbers([555,901,482,1771]))  # -> 1


def maxProfit(prices: List[int]) -> int:
    min_price = prices[0]
    max_profit = 0

    for price in prices:
        # Verificando se o valor atual é o menor já visto:
        if price < min_price:
            min_price = price

        # Subtraindo o preço atual pelo menor preço já visto para encontrar o profit atual:
        profit = price - min_price

        # Verificando se o profit atual é maior do que o maior profit já visto:
        if profit > max_profit:
            max_profit = profit

    return max_profit


# print(maxProfit([7,1,5,3,6,4]))  # -> 5
# print(maxProfit([7,6,4,3,1]))    # -> 0


def sumEvenAfterQueries(nums: List[int], queries: List[List[int]]) -> List[int]:
    result = []

    sum_actual = sum(num for num in nums if num % 2 == 0)

    for val, index in queries:
        if (nums[index] % 2) == 0:
            sum_actual -= nums[index]  # Removo o valor do sum se ele era inicialmente par.

        nums[index] = nums[index] + val

        if (nums[index] % 2) == 0:
            sum_actual += nums[index]  # Adiciona a soma se ele se tornou/continuou sendo par.

        result.append(sum_actual)

    return result


# print(sumEvenAfterQueries([1,2,3,4], [[1,0],[-3,1],[-4,0],[2,3]]))  # -> [8,6,2,4]
# print(sumEvenAfterQueries([1], [[4,0]]))                            # -> [0]


def wordSubsets(words1: List[str], words2: List[str]) -> List[str]:
    result = []

    # Criando um hash para cada palavra dentro de words2 e adicionando em temp_words:
    # Exemplo: [{'e': 1}, {'o': 1}]
    temp_words = []
    for word in words2:
        temp_hash = dict()
        for letter in word:
            temp_hash[letter] = temp_hash.get(letter, 0) + 1
        temp_words.append(temp_hash.copy())
        temp_hash.clear()

    print(temp_words)

    # Criando um hash de letras únicas do words2, onde cada letra tem um contador de aparição por item da lista:
    # O contador de aparição armazena o maior número de vezes que a letra apareceu em uma única palavra.
    # ["lo","eo"]    -> {'l': 1, 'o': 1, 'e': 1}
    # ["c","cc","b"] -> {'c': 2, 'b': 1}
    hash_word = dict()
    for item in temp_words:
        for key in item.keys():
            hash_word[key] = max(item[key], hash_word.get(key, 0))

    print(hash_word)

    # Para cada palavra em words1, transformo a palavra em um hash de letras:
    for palavra in words1:
        hash_temp = dict()

        for letter in palavra:
            hash_temp[letter] = hash_temp.get(letter, 0) + 1

        # Verifico se o número de letras únicas do hash_word é maior que o número das respectivas letras no meu hash de letras do words1.
        # Se for maior eu interrompo o loop e descarto a palavra alvo.
        # Se por fim tudo der certo (todas as letras de hash_word existirem na palavra alvo, eu adiciono a palavra em result.
        temp_value = True
        for key in hash_word.keys():
            if hash_word[key] > hash_temp.get(key, False):
                temp_value = False
                break

        if temp_value:
            result.append(palavra)
            continue

    return result


# print(wordSubsets(["amazon","apple","facebook","google","leetcode"], ["e","o"]))    # -> ["facebook","google","leetcode"]
# print(wordSubsets(["amazon","apple","facebook","google","leetcode"], ["lc","eo"]))  # -> ["leetcode"]
# print(wordSubsets(["acaac","cccbb","aacbb","caacc","bcbbb"], ["c","cc","b"]))       # -> ["cccbb"]
# print(wordSubsets(["amazon","apple","facebook","google","leetcode"], ["lo","eo"]))  # -> ["google","leetcode"]


def intToRoman(num: int) -> str:
    options = {'1': 'I', '4': 'IV', '5': 'V', '9': 'IX', '10': 'X', '40': 'XL', '50': 'L', '90': 'XC',
               '100': 'C', '400': 'CD', '500': 'D', '900': 'CM', '1000': 'M'}
    result = []

    cont = 0
    temp_fix = len(str(num))
    temp = temp_fix - 1
    while True:
        if cont == temp_fix:
            break

        actual_value = str(num)[cont] + ('0' * temp)
        ini_div = '1' + ('0' * (len(actual_value) - 1))
        division_result = int(actual_value) // int(ini_div)

        if division_result <= 3:
            target_hash = options[ini_div]
            for c in range(division_result):
                result.append(target_hash)

        elif division_result == 4 or division_result == 9:
            target_hash = options[actual_value]
            result.append(target_hash)

        elif 5 <= division_result < 9:
            temp_value = '5' + ('0' * temp)
            result.append(options[temp_value])

            temp_value = '1' + ('0' * temp)
            target_hash = options[temp_value]
            for c in range(division_result - 5):
                result.append(target_hash)

        cont += 1
        temp -= 1

    return ''.join(result)


# print(intToRoman(3749))  # -> MMMDCCXLIX
# print(intToRoman(58))    # -> LVIII
# print(intToRoman(1994))  # -> MCMXCIV


def numEquivDominoPairs(dominoes: List[List[int]]) -> int:
    new_dominoes = dict()

    for dominoe in dominoes:
        temp = tuple(sorted(dominoe))
        new_dominoes[temp] = new_dominoes.get(temp, 0) + 1

    total = 0
    for item in new_dominoes:
        if new_dominoes[item] > 1:
            total += (new_dominoes[item] * (new_dominoes[item] - 1)) // 2

    return total


# print(numEquivDominoPairs([[1,2],[2,1],[3,4],[5,6]]))                          # -> 1
# print(numEquivDominoPairs([[1,2],[1,2],[1,1],[1,2],[2,2]]))                    # -> 3
# print(numEquivDominoPairs([[2,1],[1,2],[1,2],[1,2],[2,1],[1,1],[1,2],[2,2]]))  # -> 15
# print(numEquivDominoPairs([[1,1],[2,2],[1,1],[1,2],[1,2],[1,1]]))              # -> 4


def equalFrequency(word: str) -> bool:
    letters = [letter for letter in word]
    infos = defaultdict(int)

    for letter in letters:
        infos[letter] += 1

    for key, value in infos.items():
        temp_dict = infos.copy()

        if temp_dict[key] == 1:
            del temp_dict[key]
        else:
            temp_dict[key] = temp_dict[key] - 1

        if len(set(temp_dict.values())) == 1:
            return True

        temp_dict[key] += 1

    return False


# print(equalFrequency("abcc"))  # -> True
# print(equalFrequency("aazz"))  # -> False


def buildArray(nums: List[int]) -> List[int]:
    result = [0] * len(nums)

    for indice, valor in enumerate(nums):
        result[indice] = nums[valor]

    return result


# print(buildArray([0,2,1,5,3,4]))  # -> [0,1,2,4,5,3]
# print(buildArray([5,0,1,2,3,4]))  # -> [4,5,0,1,2,3]


def addTwoNumbers(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    l1_new = []
    l2_new = []

    while l1:
        l1_new.insert(0, str(l1.val))
        l1 = l1.next

    while l2:
        l2_new.insert(0, str(l2.val))
        l2 = l2.next

    temp = int(''.join(l1_new)) + int(''.join(l2_new))
    new_temp = []

    for c in str(temp):
        new_temp.insert(0, c)

    final_result = ListNode(int(new_temp[0]))
    ini_final_result = final_result

    for value in new_temp[1:]:
        final_result.next = ListNode(int(value))
        final_result = final_result.next

    return ini_final_result


# print(addTwoNumbers(l1 = [2,4,3], l2 = [5,6,4]))            # -> [7,0,8]
# print(addTwoNumbers(l1 = [0], l2 = [0]))                    # -> [0]
# print(addTwoNumbers(l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]))  # -> [8,9,9,9,0,0,0,1]


def scoreOfString(s: str) -> int:
    string_list = [x for x in s]
    result = []

    past = 0
    for indice in range(1, len(string_list)):
        result.append(abs(ord(string_list[past]) - ord(string_list[indice])))
        past += 1

    return sum(result)


# print(scoreOfString("hello"))  # -> 13
# print(scoreOfString("zaz"))    # -> 50


def defangIPaddr(address: str) -> str:
    temp = [x for x in address]

    for indice, value in enumerate(temp.copy()):
        if value == '.':
            temp[indice] = '[.]'

    return ''.join(temp)


# print(defangIPaddr("1.1.1.1"))       # -> "1[.]1[.]1[.]1"
# print(defangIPaddr("255.100.50.0"))  # -> "255[.]100[.]50[.]0"


def numIdenticalPairs(nums: List[int]) -> int:
    temp = defaultdict(int)

    for num in nums:
        temp[num] += 1

    result = 0
    for value in temp.values():
        if value > 1:
            result += (value * (value - 1)) // 2

    return result


# print(numIdenticalPairs([1,2,3,1,1,3]))  # -> 4
# print(numIdenticalPairs([1,1,1,1]))      # -> 6
# print(numIdenticalPairs([1,2,3]))        # -> 0


def theMaximumAchievableX(num: int, t: int) -> int:
    return (num + t) + t


# print(theMaximumAchievableX(num = 4, t = 1))  # -> 6
# print(theMaximumAchievableX(num = 3, t = 2))  # -> 7


def transformArray(nums: List[int]) -> List[int]:
    for indice, num in enumerate(nums.copy()):
        if (num % 2) == 0:
            nums[indice] = 0
        else:
            nums[indice] = 1

    return sorted(nums)


# print(transformArray([4,3,2,1]))    # -> [0,0,1,1]
# print(transformArray([1,5,1,4,2]))  # -> [0,0,1,1,1]


def sortTheStudents(score: List[List[int]], k: int) -> List[List[int]]:
    score.sort(key=lambda x: x[k], reverse=True)
    return score


# print(sortTheStudents([[10,6,9,1],[7,5,11,2],[4,8,3,15]], 2))  # -> [[7,5,11,2],[10,6,9,1],[4,8,3,15]]
# print(sortTheStudents([[3,4],[5,6]], 0))                       # -> [[5,6],[3,4]]


def shuffle(nums: List[int], n: int) -> List[int]:
    x = nums[0:n]
    y = nums[n:]
    result = []

    for i in range(len(nums) // 2):
        result.append(x[i])
        result.append(y[i])

    return result


# print(shuffle([2,5,1,3,4,7], 3))      # -> [2,3,5,4,1,7]
# print(shuffle([1,2,3,4,4,3,2,1], 4))  # -> [1,4,2,3,3,2,4,1]
# print(shuffle([1,1,2,2], 2))          # -> [1,2,1,2]


import heapq

class SeatManager:
    def __init__(self, n: int):
        self.seats = list(range(1, n+1))
        heapq.heapify(self.seats)

    def reserve(self) -> int:
        return heapq.heappop(self.seats)

    def unreserve(self, seatNumber: int) -> None:
        heapq.heappush(self.seats, seatNumber)


# ["SeatManager", "reserve", "reserve", "unreserve", "reserve", "reserve", "reserve", "reserve", "unreserve"]
# teste = SeatManager(5)
# teste.reserve()
# teste.reserve()
# teste.unreserve(2)
# teste.reserve()
# teste.reserve()
# teste.reserve()
# teste.reserve()
# teste.unreserve(5)


def minSum(nums1: List[int], nums2: List[int]) -> int:
    if sum(nums1) > sum(nums2):
        biggest_nums = sum(nums1)
        smallest_nums = sum(nums2)
        biggest_array = nums1.copy()
        smallest_array = nums2.copy()
    else:
        biggest_nums = sum(nums2)
        smallest_nums = sum(nums1)
        biggest_array = nums2.copy()
        smallest_array = nums1.copy()

    if (biggest_nums == smallest_nums) and (biggest_array.count(0) == 0 and smallest_array.count(0) == 0):
        return biggest_nums

    if (biggest_nums == smallest_nums) and (biggest_array.count(0) == smallest_array.count(0)):
        return biggest_array.count(0)

    target_value = biggest_nums + biggest_array.count(0)

    temp = smallest_array.count(0)
    if temp > 0:
        diference_target = (target_value - smallest_nums) / temp
    else:
        return -1

    if diference_target >= 1:
        return target_value
    else:
        if biggest_array.count(0) == 0:
            return -1
        else:
            return smallest_nums + smallest_array.count(0)


# print(minSum([3,2,0,1,0], [6,5,0]))                                                                # -> 12
# print(minSum([2,0,2,0], [1,4]))                                                                    # -> -1
# print(minSum([8,13,15,18,0,18,0,0,5,20,12,27,3,14,22,0], [29,1,6,0,10,24,27,17,14,13,2,19,2,11]))  # -> 179
# print(minSum([9,5], [15,12,5,21,4,26,27,9,6,29,0,18,16,0,0,0,20]))                                 # -> -1
# print(minSum([1,2,3,2], [1,4,3]))                                                                  # -> 8
# print(minSum([0], [0]))                                                                            # -> 1


def threeConsecutiveOdds(arr: List[int]) -> bool:
    count = 0

    for c in arr:
        if (c % 2) == 1:
            count += 1
        else:
            count = 0

        if count == 3:
            return True

    return False


# print(threeConsecutiveOdds([2,6,4,1]))               # -> False
# print(threeConsecutiveOdds([1,2,34,3,4,5,7,23,12]))  # -> True


def reverse(x: int) -> int:
    temp = []
    string_value = str(x)

    for c in string_value[len(string_value): 0: -1]:
        temp.append(c)

    if string_value[0] != '-':
        temp.append(string_value[0])
    else:
        temp.insert(0, string_value[0])

    result = int(''.join(temp))

    if result <= (2 ** 31) - 1 and result >= -2 ** 31:
        return result

    return 0


# print(reverse(123))         # -> 321
# print(reverse(-123))        # -> -321
# print(reverse(120))         # -> 21
# print(reverse(1534236469))  # -> 0


def swapPairs(head: Optional[ListNode]) -> Optional[ListNode]:
    if not head or head.next is None:
        return head

    temp = ListNode()
    ini_temp = temp

    while head:
        next_node = head.next

        if next_node is None:
            temp.next = ListNode(head.val)
            break

        temp.next = ListNode(head.next.val)
        temp = temp.next
        temp.next = ListNode(head.val)
        temp = temp.next

        head = head.next.next


    return ini_temp.next


# print(swapPairs([1,2,3,4]))  # -> [2,1,4,3]
# print(swapPairs([]))         # -> []
# print(swapPairs([1]))        # -> [1]
# print(swapPairs([1,2,3]))    # -> [2,1,3]


def findEvenNumbers(digits: List[int]) -> List[int]:
    from itertools import permutations

    result = set()

    for i, j, k in permutations(range(len(digits)), 3):
        # t = (str(digits[i]), str(digits[j]), str(digits[k]))
        d1, d2, d3 = (digits[i], digits[j], digits[k])

        if d1 == 0 :
            continue

        # temp = list(t)
        # result.add(int(''.join(temp)))

        num = d1*100 + d2*10 + d3*1

        if (num % 2) == 1:
            continue

        result.add(num)

    return sorted(list(result))


# print(findEvenNumbers([2,1,3,0]))    # -> [102,120,130,132,210,230,302,310,312,320]
# print(findEvenNumbers([2,2,8,8,2]))  # -> [222,228,282,288,822,828,882]
# print(findEvenNumbers([3,7,5]))      # -> []


def letterCombinations(digits: str) -> List[str]:
    from itertools import product

    if not digits:
        return []

    letters_translate = {
        '2': ['a', 'b', 'c'],
        '3': ['d', 'e', 'f'],
        '4': ['g', 'h', 'i'],
        '5': ['j', 'k', 'l'],
        '6': ['m', 'n', 'o'],
        '7': ['p', 'q', 'r', 's'],
        '8': ['t', 'u', 'v'],
        '9': ['w', 'x', 'y', 'z']
    }

    if len(digits) == 1:
        return letters_translate[digits[0]]

    array_letters = [letters_translate[d] for d in digits]

    result = []

    for j in product(*array_letters):
        result.append(''.join(j))

    return result


# print(letterCombinations("23"))  # -> ["ad","ae","af","bd","be","bf","cd","ce","cf"]
# print(letterCombinations(""))    # -> []
# print(letterCombinations("2"))   # -> ["a","b","c"]


def isSameTree(p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
    if not p and not q:
        return True
    if not p or not q:
        return False
    if p.val != q.val:
        return False

    return isSameTree(p.left, q.left) and isSameTree(p.right, q.right)


# print(isSameTree(p = [1,2,3], q = [1,2,3]))   # -> True
# print(isSameTree(p = [1,2], q = [1,None,2]))  # -> False
# print(isSameTree(p = [1,2,1], q = [1,1,2]))   # -> False


def triangleType(nums: List[int]) -> str:
    validation = nums[0] + nums[1] > nums[2] and nums[0] + nums[2] > nums[1] and nums[1] + nums[2] > nums[0]
    if not validation:
        return "none"

    temp = len(set(nums))
    if temp == 1:
        return "equilateral"
    elif temp == 2:
        return "isosceles"

    return "scalene"


print(triangleType([3,3,3]))  # -> "equilateral"
print(triangleType([3,4,5]))  # -> "scalene"
