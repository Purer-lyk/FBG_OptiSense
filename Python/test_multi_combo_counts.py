"""Exact simultaneous-combination counts, not marginal position counts."""
import json
from unittest.mock import patch
import test_multi_touch_capture as multi

class MultiComboCountsTests(multi.MultiTouchTests):
    def select(self, *labels):
        self.page.select_no_press()
        for label in labels: self.page.select_grid_point(label)

    def test_numeric_sort_and_dedup_match_saved_labels(self):
        p=self.page; p.mode_combo.setCurrentIndex(1); self.select(10,2)
        self.assertTrue(p.capture_current_press())
        self.assertEqual(p.combo_counts,{(2,10):1})
        self.assertIn('{2, 10}',p.selected_label.text())
        self.assertEqual(json.loads((p.session_dir/'frames.jsonl').read_text('utf-8'))['quality']['position_labels'],[2,10])

    def test_undo_audit_failure_still_reflects_actual_writer_undo(self):
        p=self.page; p.mode_combo.setCurrentIndex(1); self.select(22,23)
        self.assertTrue(p.capture_current_press())
        with patch.object(p.writer,'write_event',side_effect=OSError('audit failed')):
            self.assertTrue(p.undo_last_annotation(confirm=False))
        self.assertEqual(p.combo_counts,{})
        self.assertEqual(len(p.capture_history),0)
        self.assertIn('审计',p.capture_status_label.text())
        self.assertEqual((p.session_dir/'frames.jsonl').read_text('utf-8'),'')
        self.assertEqual(json.loads((p.session_dir/'combination_counts.json').read_text('utf-8'))['active_samples_count'],0)

    def test_rejected_frame_does_not_count_and_reports_failure(self):
        p=self.page; p.mode_combo.setCurrentIndex(1); self.select(22,23)
        p._ensure_session()
        with patch.object(p.writer,'write_frame',side_effect=BufferError('rejected')):
            self.assertFalse(p.capture_current_press())
        self.assertEqual(p.combo_counts,{})
        self.assertEqual(p.capture_history,[])
        self.assertEqual(p.sample_index,0)
        self.assertIn('失败',p.capture_status_label.text())
        self.assertTrue(p.capture_current_press())
        self.assertEqual(p.combo_counts,{(22,23):1})

    def test_snapshot_failure_preserves_previous_json_and_reports_not_fake_success(self):
        p=self.page; p.mode_combo.setCurrentIndex(1); self.select(22,23)
        self.assertTrue(p.capture_current_press()); root=p.session_dir
        before=(root/'combination_counts.json').read_bytes()
        with patch('manual_stress_capture.os.replace',side_effect=OSError('readonly')):
            self.assertTrue(p.capture_current_press())
            self.assertIn('统计保存失败',p.capture_status_label.text())
            self.assertFalse(p.close_session())
        self.assertEqual((root/'combination_counts.json').read_bytes(),before)
        self.assertEqual(p.combo_counts,{(22,23):2})
        self.assertTrue(p.close_session())
        self.assertEqual(json.loads((root/'combination_counts.json').read_text('utf-8'))['active_samples_count'],2)

    def test_order_independent_current_hint_table_empty_undo_and_session(self):
        p=self.page; p.mode_combo.setCurrentIndex(1)
        self.select(23,22)
        self.assertIn('当前组合已采集 0 组',p.selected_label.text())
        self.assertTrue(p.capture_current_press())
        self.select(22,23); self.assertTrue(p.capture_current_press())
        self.assertIn('当前组合已采集 2 组',p.selected_label.text())
        self.select(1,13,25); self.assertTrue(p.capture_current_press())
        self.select(); self.assertTrue(p.capture_current_press())
        self.assertEqual(p.combo_counts,{(22,23):2,(1,13,25):1,():1})
        self.assertEqual(p.combo_counts_table.rowCount(),3)
        root=p.session_dir
        snapshot=json.loads((root/'combination_counts.json').read_text('utf-8'))
        self.assertEqual(snapshot['active_samples_count'],4)
        self.assertEqual(snapshot['counting_semantics'],'successful45pointframes_minusundone')
        self.select(23,22)
        self.assertIn('当前组合已采集 2 组',p.selected_label.text())
        self.assertTrue(p.undo_last_annotation(confirm=False))
        self.assertEqual(p.combo_counts,{(22,23):2,(1,13,25):1})
        self.assertTrue(p.undo_last_annotation(confirm=False))
        self.assertEqual(p.combo_counts,{(22,23):2})
        self.assertEqual(p.capture_counts[22],2)
        p.writer.flush()
        self.assertEqual(len((root/'frames.jsonl').read_text('utf-8').splitlines()),2)
        snapshot=json.loads((root/'combination_counts.json').read_text('utf-8'))
        self.assertEqual(snapshot['active_samples_count'],2)
        self.assertEqual(snapshot['combinations'],[{'position_labels':[22,23],'count':2}])
        p.close_session()
        self.assertEqual(p.combo_counts,{})
        self.assertIn('当前组合已采集 0 组',p.selected_label.text())
        self.assertEqual(json.loads((root/'combination_counts.json').read_text('utf-8'))['active_samples_count'],2)
        self.assertTrue(p.capture_current_press())
        self.assertNotEqual(p.session_dir,root)
        self.assertEqual(p.combo_counts,{(22,23):1})

if __name__=='__main__':
    import unittest
    unittest.main()
