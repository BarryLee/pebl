import unittest

from pebl.learner.classifier import LocalCPDCache
from pebl.learner.wrapper import SharedLocalCPDCache

class TestLocalCPDCache(unittest.TestCase):

    def setUp(self):
        self.l_ = LocalCPDCache()

    def testCallLocalCPDCache(self):
        self.l_('a',1)
        self.l_.setdefault('b',2)
        assert self.l_('a',2) == 1
        assert self.l_.setdefault('b',4) == 2
        assert self.l_('c',3) == self.l_.setdefault('c')
        assert self.l_('d',4) == 4
        
class TestSharedCPDCache(unittest.TestCase):

    def setUp(self):
        self._l = LocalCPDCache()
        self._l((0,4),1)
        self._l((1,4),2)
    
    def testAll(self):
        sl1 = SharedLocalCPDCache(self._l, [2,4])
        sl2 = SharedLocalCPDCache(self._l, [1,3,4])

        assert sl2((0,2)) == 2
        sl1((0,1),1)
        sl2((1,2),2)
        sl1((0,(1,1)), 3)
        assert sl1((0,1)) == self._l((2,4))
        assert sl2((1,2)) == self._l((3,4))
        assert sl1((0,(1,1))) == self._l((2,(4,4)))
        self.assertRaises(IndexError, sl1, (0,1,2),3)

if __name__ == '__main__':
    unittest.main()
