from unittest import TestCase
from unittest.mock import patch

from utility import util


class TestUtil(TestCase):

    @patch("util.random_segmentation")
    def test_apply_mask_to_input_text_1(self, mock_random):
        text = "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with the ends " \
               "of worms and an oozy smell, nor yet a dry, bare, sandy hole with nothing in it to sit down on or " \
               "to eat: it was a hobbit'hole, and that means comfort"
        mock_random.side_effect = [[1, 4], [9, 38]]
        result_input, result_output = util.apply_mask_to_input_text(text)
        expected_input = "In a hole in the ground there lived a extra_id_0 Not a nasty, dirty, wet hole, filled with the " \
                         "ends of worms and an oozy smell, nor yet a dry, bare, sandy hole with nothing in it to sit down" \
                         " on or to eat: it was a hobbit'hole, extra_id_1"
        expected_output = 'extra_id_0 hobbit. extra_id_1 and that means comfort'
        self.assertEqual(expected_input, result_input)
        self.assertEqual(expected_output, result_output)

    @patch("util.random_segmentation")
    def test_apply_mask_to_input_text_2(self, mock_random):
        text = "In a hole in the ground there lived a hobbit."
        mock_random.side_effect = [[1], [9]]
        result_input, result_output = util.apply_mask_to_input_text(text)
        expected_input = 'In a hole in the ground there lived a extra_id_0'
        expected_output = 'extra_id_0 hobbit.'
        self.assertEqual(expected_input, result_input)
        self.assertEqual(expected_output, result_output)

    @patch("util.random_segmentation")
    def test_apply_mask_to_input_text_3(self, mock_random):
        text = "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with the ends " \
               "of worms."
        mock_random.side_effect = [[2], [20]]
        result_input, result_output = util.apply_mask_to_input_text(text)
        expected_input = 'In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole,' \
                         ' filled with the ends extra_id_0'
        expected_output = 'extra_id_0 of worms.'
        self.assertEqual(expected_input, result_input)
        self.assertEqual(expected_output, result_output)

    @patch("util.random_segmentation")
    def test_apply_mask_to_input_text_4(self, mock_random):
        text = "Now they laid Boromir in the middle of the boat that was to bear him away. The grey hood and " \
               "elven-cloak they folded and placed beneath his head. They combed his long dark hair and arrayed it" \
               " upon his shoulders. The golden belt of L rien gleamed about his waist. His helm they set beside " \
               "him, and across his lap they laid the cloven horn and the hilts and shards of his sword; beneath his" \
               " feet they put the swords of his enemies. Then fastening the prow to the stern of the other boat, " \
               "they drew him out into the water. They rowed sadly along the shore, and turning into the " \
               "swift-running channel they passed the green sward of Parth Galen. The steep sides of Tol Brandir " \
               "were glowing: it was now mid-afternoon. As they went south the fume of Rauros rose and shimmered " \
               "before them, a haze of gold. The rush and thunder of the falls shook the windless air."
        mock_random.side_effect = [[1, 5, 1, 2, 7], [9, 9, 64, 24, 39]]
        result_input, result_output = util.apply_mask_to_input_text(text)
        expected_input = \
                   "Now they laid Boromir in the middle of the extra_id_0 that was to bear him away. The grey " \
                   "hood extra_id_1 placed beneath his head. They combed his long dark hair and arrayed it upon " \
                   "his shoulders. The golden belt of L rien gleamed about his waist. His helm they set beside him," \
                   " and across his lap they laid the cloven horn and the hilts and shards of his sword; beneath" \
                   " his feet they put the swords of his enemies. Then fastening the prow to extra_id_2 stern of " \
                   "the other boat, they drew him out into the water. They rowed sadly along the shore, and turning " \
                   "into the swift-running channel extra_id_3 the green sward of Parth Galen. The steep sides of " \
                   "Tol Brandir were glowing: it was now mid-afternoon. As they went south the fume of Rauros rose " \
                   "and shimmered before them, a haze of gold. The rush and thunder extra_id_4"
        expected_output = 'extra_id_0 boat extra_id_1 and elven-cloak they folded and extra_id_2 the extra_id_3 they passed' \
                          ' extra_id_4 of the falls shook the windless air.'
        self.assertEqual(expected_input, result_input)
        self.assertEqual(expected_output, result_output)

    def test_apply_mask_to_input_text_5(self):
        text = "The Great River one"

        with self.assertRaises(ValueError):
            util.apply_mask_to_input_text(text)
