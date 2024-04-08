# Number of Students Unable to Eat Lunch LeetCode Easy
# https://leetcode.com/problems/number-of-students-unable-to-eat-lunch/description/
def countStudents(self, students: List[int], sandwiches: List[int]) -> int:
    students = collections.deque(students)
    length = len(students)
    curr_sandwhich = 0
    while students and curr_sandwhich < len(sandwiches) and length:
        startLen = len(students)
        if students[0] != sandwiches[curr_sandwhich]:
            students.append(students.popleft())
            length -= 1
        else:
            students.popleft()
            curr_sandwhich += 1
            length = len(students)
    return len(students)