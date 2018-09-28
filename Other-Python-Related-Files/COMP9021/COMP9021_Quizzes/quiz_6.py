# Defines two classes, Point() and Disk().
# The latter has an "area" attribute and three methods:
# - change_radius(r)
# - intersects(disk), that returns True or False depending on whether
#   the disk provided as intersects the disk object.
# - absorb(disk), that returns a new disk object that represents the smallest
#   disk that contains both the disk provided as argument and the disk object.
#
# Written by *** and Eric Martin for COMP9021


from math import pi, hypot


class Point():
    def __init__(self, x = 0, y = 0):
        self.x = x
        self.y = y
    def __repr__(self):
        return 'Point({:.2f}, {:.2f})'.format(self.x, self.y)

class Disk():
    def __init__(self, *, centre = Point(), radius = 0):
        self.centre = centre
        self.radius = radius
        self.area = pi * (self.radius ** 2)

    def change_radius(self, r):
        self.radius = r
        self.area = pi * (self.radius ** 2)

    def intersects(self, disk2):
        self.disk2 = disk2
        self.r1 = self.radius
        self.r2 = self.disk2.radius
        if hypot((self.disk2.centre.y - self.centre.y), (self.disk2.centre.x - self.centre.x))\
                <= abs(self.r1 + self.r2):
            return True
        else:
            return False

    def absorb(self, disk3):
        self.disk3 = disk3

        if not abs(self.disk3.radius - self.radius) < hypot((self.disk3.centre.y - self.centre.y), (self.disk3.centre.x - self.centre.x)):
            if self.disk3.radius >= self.radius:
                return self.disk3
            else:
                return self
        else:
            self.r = (hypot((self.disk3.centre.y - self.centre.y), (self.disk3.centre.x - self.centre.x)) + self.radius + self.disk3.radius) / 2
            self.xx = (self.r - self.radius) * (self.disk3.centre.x - self.centre.x) / (2 * self.r - self.disk3.radius
                - self.radius) + self.centre.x
            self.yy = (self.r - self.radius) * (self.disk3.centre.y - self.centre.y) / (2 * self.r - self.disk3.radius
                - self.radius) + self.centre.y
            return Disk(centre = Point(self.xx, self.yy), radius = self.r)

    def __repr__(self):
        return 'Disk(Point({:.2f}, {:.2f}), {:.2f})'.format(self.centre.x, self.centre.y, self.radius)








