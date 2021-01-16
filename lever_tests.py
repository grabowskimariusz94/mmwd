import unittest
import numpy as np
import lever as lv


class LeverTest(unittest.TestCase):
	def test_objectiveFunc(self):
		func = lv.objectiveFunc(Sol=[0,3,5,2,7,0,0,0,4,2],MS=[],M=50,g=10,R=5)
		self.assertEqual(func,170)

		func = lv.objectiveFunc(Sol=[0,3,5,2,7,46,0,0,0,4,2],MS=[],M=50,g=10,R=5)
		self.assertEqual(func,170)

	def test_sortBestSol(self):
		self.assertEqual(
                lv.sortBestSol(S=[[1,1,1,0,0,0],[0,0,0,1,1,1]],MS=[1,1,1],M=10,g=10,R=3),\
                ([70,50],[1,0]))

	def test_transformSol(self):
		self.assertEqual(lv.transformSol([[None,2,3,None]],[6,7,8,9]),[[0,8,9,0]])

	def test_sick(self):
		self.assertTrue(lv.sick(kid=[1,0,0,0,6,3,4,4],MS=np.array([3,3,3,4,4,4])))
		self.assertFalse(lv.sick(kid=[1,0,0,0,6,3,4,4],MS=np.array([1,1,3,3,4,4,6,6])))

	def test_rotateMutation(self):
		self.assertEqual(lv.rotateMutation([[7,8,9]]),[[9,7,8]])

	def test_cannotSolve(self):
		self.assertTrue(lv.cannotSolve(MS=[5,6,7],M=741,g=10,R=5))
		self.assertFalse(lv.cannotSolve(MS=[5,6,7],M=740,g=10,R=5))


if __name__ == '__main__':
    unittest.main()

