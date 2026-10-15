#returns 2nd distinct largest in a list of numbers

def second_largest(nums):
    first = None
    second = None
    for x in nums:
        if first is None or x > first:
            second = first
            first = x
        elif x != first and (second is None or x > second):
            second = x

    return second

nums = [2,4,3,5,5,6]

print(second_largest(nums))