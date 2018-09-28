from collections import deque
from math import hypot


# Q1
def available_coloured_pieces(file):
    pieces_list = {}
    for i in file:
        start = i.find('M')
        if start != -1:
            mid = i.find('z')
            section = i[start + 1: mid]
            section = section.split('L')
            color_section = i[mid + 9: -4]
            if color_section in pieces_list:
                return False
            improved_list = []
            for j in section:
                improved_list.append(tuple(map(float, j.split())))
            pieces_list[color_section] = improved_list
    return pieces_list


def are_valid(coloured_pieces):
    if not coloured_pieces:
        # if two shapes in the same color.
        # print('False0')
        return False
    for section in coloured_pieces.values():
        if not len(section) >= 3:
            # print("false1")
            return False
            # if not coincide
        if not len(set(section)) == len(section):
            # print("false2")
            return False
        # next to determine whether it is a convex polygon.
        for j in section[2:]:
            vertex3 = j
            vertex2 = section[section.index(j) - 1]
            vertex1 = section[section.index(j) - 2]
            # if three point on a straight line.
            if len(section) == 3:
                if (vertex2[1] - vertex3[1]) * (vertex1[0] - vertex2[0]) == (vertex2[0] - vertex3[0]) * (
                            vertex1[1] - vertex2[1]):
                    # print('false3')
                    return False
                    # if n > 3, see if it is a convex polygon.
            else:
                t = vertex3[0] * (vertex2[1] - vertex1[1]) + vertex3[1] * (vertex1[0] - vertex2[0]) - vertex1[0] * \
                                                                                                      vertex2[1] + \
                    vertex2[0] * vertex1[1]
                for k in section[3:]:
                    next_vertex = k
                    if t > 0 > (next_vertex[0] * (vertex2[1] - vertex1[1]) + next_vertex[1] * (vertex1[0] - vertex2[0])
                                    - vertex1[0] * vertex2[1] + vertex2[0] * vertex1[1]):
                        # print('false4')
                        return False
                    elif t < 0 < (next_vertex[0] * (vertex2[1] - vertex1[1]) + next_vertex[1] * (
                                vertex1[0] - vertex2[0])
                                      - vertex1[0] * vertex2[1] + vertex2[0] * vertex1[1]):
                        # print('false5')
                        return False
                    elif t == (next_vertex[0] * (vertex2[1] - vertex1[1]) + next_vertex[1] * (vertex1[0] - vertex2[0])
                                   - vertex1[0] * vertex2[1] + vertex2[0] * vertex1[1]) == 0:
                        # print('false6')
                        return False
    return True


# Q2
# this is to calculate each length of edges of one shape.
def calculate_lenth(shape):
    lenth_list = []
    for i in shape[: -1]:
        vertex1 = i
        vertex2 = shape[shape.index(i) + 1]
        lenth_of_side = hypot(vertex2[0] - vertex1[0], vertex2[1] - vertex1[1])
        lenth_list.append(lenth_of_side)
    lenth_list.append(hypot(shape[-1][0] - shape[0][0], shape[-1][1] - shape[0][1]))
    return lenth_list


# this function is created for calculating cos values of angles.
def distance_of_two_points(point1, point2):
    return hypot(point1[0] - point2[0], point1[1] - point2[1])


def calculate_angles(shape):
    angle_list = []
    for i in shape[: -2]:
        vertex1 = i
        vertex2 = shape[shape.index(i) + 1]
        vertex3 = shape[shape.index(i) + 2]
        vector1 = (vertex2[0] - vertex1[0], vertex2[1] - vertex1[1])
        vector2 = (vertex3[0] - vertex2[0], vertex3[1] - vertex2[1])
        angle_cos = (vector1[0] * vector2[0] + vector1[1] * vector2[1]) / (
            distance_of_two_points(vertex2, vertex1) * distance_of_two_points(vertex3, vertex2))
        angle_list.append(angle_cos)
    # the last angle and the first angle. (Traversing cannot reach these guys.)
    vertex11 = shape[-2]
    vertex22 = shape[-1]
    vertex33 = shape[0]
    vertex44 = shape[1]
    vector11 = (vertex22[0] - vertex11[0], vertex22[1] - vertex11[1])
    vector22 = (vertex33[0] - vertex22[0], vertex33[1] - vertex22[1])
    vector44 = (vertex44[0] - vertex33[0], vertex44[1] - vertex33[1])
    angle_list.append((vector11[0] * vector22[0] + vector11[1] * vector22[1]) / (
        distance_of_two_points(vertex22, vertex11) * distance_of_two_points(vertex33, vertex22)))
    angle_list.append((vector22[0] * vector44[0] + vector22[1] * vector44[1]) / (
        distance_of_two_points(vertex33, vertex22) * distance_of_two_points(vertex44, vertex33)))
    return angle_list


# this function is to check the attribute of two sets of pieces,
# which can be used for checking both lengths and angles.
def check_attributes(base_list, list):
    battle_list_clockwise = deque(list)
    battle_list_anticlockwise = deque(list[:: -1])
    deque_base_list = deque(base_list)
    for _ in range(len(list) - 1):
        if battle_list_clockwise == deque_base_list or battle_list_anticlockwise == deque_base_list:
            return True
        else:
            battle_list_clockwise.append(battle_list_clockwise.popleft())
            battle_list_anticlockwise.append(battle_list_anticlockwise.popleft())
    if battle_list_clockwise == deque_base_list or battle_list_anticlockwise == deque_base_list:
        return True
    else:
        return False


def are_identical_sets_of_coloured_pieces(cp1, cp2):
    if (not cp1) or (not cp2):
        # if two shapes in the same color.
        return False
    if (len(cp1) != len(cp2)) or (cp1.keys() != cp2.keys()):
        return False
    # then judge if they are in the same shape.
    for color in cp1:
        if len(cp1[color]) != len(cp2[color]):
            return False
        shape_lenth1 = calculate_lenth(cp1[color])
        shape_lenth2 = calculate_lenth(cp2[color])
        shape_angle1 = calculate_angles(cp1[color])
        shape_angle2 = calculate_angles(cp2[color])
        if (not check_attributes(shape_lenth1, shape_lenth2)) and (
                not check_attributes(shape_angle1, shape_angle2)):
            return False
    return True


# Q3
# to calculate polygons' areas
def area(L):
    S = 0
    for i in range(len(L) - 1):
        S += L[i][0] * L[i + 1][1] - L[i + 1][0] * L[i][1]
    S += L[-1][0] * L[0][1] - L[0][0] * L[-1][1]
    return S / 2.0


# to check if their areas are equal.
def areas_are_eaqual(shape, pieces):
    shape_area = 0
    for one in shape.values():
        shape_area = area(one)
    pieces_area = 0
    for many in pieces.values():
        pieces_area += abs(area(many))
    if shape_area == pieces_area:
        return True
    else:
        return False


# Method: Ray casting or Even odd rule
# All vertexes of pieces should be in the shapes or on the edges(including is the same as vertexes of the shape).
def is_solution(tangram, shape):
    if not areas_are_eaqual(shape, tangram):
        return False
    for vertex_list in tangram.values():
        for piece_vertex in vertex_list:  # select each of the vertexes of pieces
            n = 0  # a counter for calculate the intersection on the left side
            for only_one_shape in shape.values():  # actually it only circulate once.
                # for the last edge which is between only_one_shape[-1] and between only_one_shape[0].
                if (piece_vertex[1] - only_one_shape[-1][1]) * (only_one_shape[0][0] - piece_vertex[0]) == (
                            piece_vertex[0] - only_one_shape[-1][0]) * (only_one_shape[0][1] - piece_vertex[1]):
                    break
                for shape_vertex in only_one_shape[:-1]:
                    # between (x1, y1) and (x2, y2), there is a edge.
                    # if piece_vertex on the edges of shape_vertex
                    # piece_vertex[0] = x, piece_vertex[1] = y.
                    # (only_one_shape.index(shape_vertex) + 1) is the index of the (x2, y2)
                    # print('x1, y1 = ', shape_vertex[0], shape_vertex[1])
                    # print('x2, y2 = ', only_one_shape[only_one_shape.index(shape_vertex) + 1][0],
                    if (piece_vertex[1] - shape_vertex[1]) * (
                                only_one_shape[only_one_shape.index(shape_vertex) + 1][0] -
                                piece_vertex[0]) == (piece_vertex[0] - shape_vertex[0]) * \
                            (only_one_shape[only_one_shape.index(shape_vertex) + 1][1] - piece_vertex[1]):
                        break
                    # at this stage, we've finished the situations of "on the edges".
                    if (not shape_vertex[1] == only_one_shape[only_one_shape.index(shape_vertex) + 1][1]) and \
                            ((shape_vertex[1] <= piece_vertex[1] <= only_one_shape[only_one_shape.index(shape_vertex) + 1][1]) or (shape_vertex[1] >= piece_vertex[1] >= only_one_shape[only_one_shape.index(shape_vertex) + 1][1])):
                        x = ((piece_vertex[1] - only_one_shape[only_one_shape.index(shape_vertex) + 1][1]) * (
                            shape_vertex[0] - only_one_shape[only_one_shape.index(shape_vertex) + 1][0])) / (
                                shape_vertex[1] - only_one_shape[only_one_shape.index(shape_vertex) + 1][1]) + \
                            only_one_shape[only_one_shape.index(shape_vertex) + 1][0]
                        if (shape_vertex[0] <= x <= only_one_shape[only_one_shape.index(shape_vertex) + 1][0] or
                                                only_one_shape[only_one_shape.index(shape_vertex) + 1][0] <=
                                                x <= shape_vertex[0]) and (x < piece_vertex[0]):
                            n += 1
                else:
                    # for the last edge which is between only_one_shape[-1] and between only_one_shape[0].
                    if (not only_one_shape[0][1] == only_one_shape[-1][1]) and \
                            (only_one_shape[0][1] <= piece_vertex[1] <= only_one_shape[-1][1]) or (only_one_shape[0][1]  >= piece_vertex[1] >= only_one_shape[-1][1]):
                        x = (((piece_vertex[1] - only_one_shape[0][1]) * (only_one_shape[-1][0] - only_one_shape[0][0])) / (only_one_shape[-1][1] - only_one_shape[0][1])) + only_one_shape[0][0]
                        if only_one_shape[-1][0] <= x <= only_one_shape[0][0] or only_one_shape[0][0] <= x <= only_one_shape[-1][0] and (x < piece_vertex[0]):
                            n += 1
                    if n % 2 == 0:
                        return False
    return True


# for checking use.
'''
# for q1
file = open('pieces_A.xml')
coloured_pieces = available_coloured_pieces(file)
print(are_valid(coloured_pieces))

file = open('pieces_AA.xml')
coloured_pieces = available_coloured_pieces(file)
print(are_valid(coloured_pieces))

file = open('incorrect_pieces_1.xml')
coloured_pieces = available_coloured_pieces(file)
print(are_valid(coloured_pieces))

file = open('incorrect_pieces_2.xml')
coloured_pieces = available_coloured_pieces(file)
print(are_valid(coloured_pieces))

file = open('incorrect_pieces_3.xml')
coloured_pieces = available_coloured_pieces(file)
print(are_valid(coloured_pieces))

file = open('incorrect_pieces_4.xml')
coloured_pieces = available_coloured_pieces(file)
print(are_valid(coloured_pieces))
print('------------------------------------------------')

# for q2
file = open('pieces_A.xml')
coloured_pieces_1 = available_coloured_pieces(file)
file = open('pieces_AA.xml')
coloured_pieces_2 = available_coloured_pieces(file)
print(are_identical_sets_of_coloured_pieces(coloured_pieces_1, coloured_pieces_2))

file = open('shape_A_1.xml')
coloured_pieces_2 = available_coloured_pieces(file)
print(are_identical_sets_of_coloured_pieces(coloured_pieces_1, coloured_pieces_2))
print('------------------------------------------------')

# for q3
file = open('shape_A_1.xml')
a = available_coloured_pieces(file)
file = open('tangram_A_1_b.xml')
b = available_coloured_pieces(file)
print(areas_are_eaqual(a, b))
print(is_solution(b, a))
file.close()
'''


