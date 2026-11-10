"""PR5 five-point migration: real selection and strict measured shape contracts."""
import copy
import unittest
import numpy as np
import temporary_test_mode as t
from test_temporary_test_mode import reference


def selected(stride=1, asymmetric=False):
    table, _, _ = reference()
    x = np.asarray([row.measured_nm for row in table])[::stride]
    y = np.full_like(x, 30.)
    for center in np.linspace(1527, 1563, 9):
        sigma = np.where(x < center, .12, .26) if asymmetric else .16
        y += 1800*np.exp(-.5*((x-center)/sigma)**2)
    return table, x, y, t.auto_select_dense_plan(table, x, y)


class FiveSelectionTests(unittest.TestCase):
    def test_actual_auto_selection_keeps_height_rule_with_two_real_pairs(self):
        from temporary_optical_units import adc_code_to_dbm
        for stride, asymmetric in ((1, False), (2, False), (1, True)):
            table, x, y, plan = selected(stride, asymmetric)
            self.assertEqual((plan['point_count'], plan['peak_count'], plan['points_per_peak']), (45,9,5))
            self.assertEqual(plan['symmetry_basis'], 'fitted_dbm_height')
            self.assertEqual(plan['selection_height_region'], 'dbm_visual_upper_half')
            self.assertEqual(t.validate_plan_payload(plan)['points_per_peak'], 5)
            for group, band in enumerate(plan['selection_band_checks']):
                rows = plan['rows'][group*5:group*5+5]
                indices = [r['index'] for r in rows]
                check = plan['symmetry_checks'][group]
                self.assertEqual(check['center_index'], indices[2])
                self.assertEqual(check['left_indices'], indices[1::-1])
                self.assertEqual(check['right_indices'], indices[3:])
                self.assertTrue(all(i%stride == 0 for i in indices))
                self.assertEqual([r['codes'] for r in rows], [list(table[i].codes) for i in indices])
                raw = adc_code_to_dbm(y[np.asarray(indices)//stride],2)
                np.testing.assert_allclose(raw, band['selected_raw_dbm'])
                self.assertTrue(np.all(raw >= band['threshold_dbm']-1e-8))
                if asymmetric:
                    self.assertNotEqual(indices[0]+indices[-1], 2*indices[2])
            np.testing.assert_array_equal(plan['reference_values'], y)
            wrong = copy.deepcopy(plan); wrong['symmetry_checks'][0]['fitted_pair_difference_db'][0] += 1
            with self.assertRaises(ValueError): t.validate_plan_payload(wrong)

class FiveWireTests(unittest.TestCase):
    def test_retired_frame_shape_fails_before_any_frame_decoding(self):
        with self.assertRaisesRegex(ValueError, '每峰五点'):
            t.decode(b'',123,0,point_count=63,points_per_peak=7)

    def test_seven_rows_per_peak_are_rejected_not_truncated(self):
        table, _, _ = reference()
        groups = [[center + offset for offset in (-6,-4,-2,0,2,4,6)] for center in range(100,1901,225)]
        rows = [dict(group=g,index=i,codes=list(table[i].codes)) for g,group in enumerate(groups,1) for i in group]
        with self.assertRaises(ValueError): t.command(rows,32,123)
        with self.assertRaises(ValueError): t.command(rows,32,123,single_sample=True)

    def test_c_firmware_accepts_five_and_rejects_seven_without_partial_publish(self):
        import ctypes, os, subprocess, tempfile
        from pathlib import Path
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir='C:/Users/Administrator/AppData/Local/hermes/cache/scratch', ignore_cleanup_errors=True) as folder:
            source=Path(folder)/'parser.c'; dll=Path(folder)/'parser.dll'
            source.write_text('''#include <string.h>
#include "candidate_route_protocol.h"
__declspec(dllexport) int parse(const unsigned char *p, unsigned short *values) {
 CandidateRoute out; memset(&out,0xA5,sizeof(out));
 int ok=CandidateRoute_Parse(p,808,&out);
 if(!ok) { const unsigned char *b=(const unsigned char *)&out;
  for(unsigned i=0;i<sizeof(out);i++) if(b[i]!=0xA5) return -1; return 0; }
 values[0]=out.pointCount; values[1]=out.pointsPerPeak;
 for(unsigned i=0;i<out.pointCount;i++) {values[2+2*i]=out.indices[i];values[3+2*i]=out.pointDelayUs[i];}
 return 1;
}
''',encoding='utf-8')
            subprocess.run(['C:/msys64/ucrt64/bin/gcc.exe','-std=c99','-Wall','-shared','-O2','-I',str(root/'JDSU/Core/Inc'),str(source),'-o',str(dll)],check=True,capture_output=True)
            lib=ctypes.CDLL(str(dll)); lib.parse.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ushort)]
            table, _, _=reference()
            for count in range(5,46,5):
                rows=[dict(group=i//5+1,index=i*2,codes=list(table[i*2].codes)) for i in range(count)]
                for channel in range(4):
                    for single in (False,True):
                        for wait in (50,850,15000):
                            packet=t.command(rows,32,123,adc_channel=channel,single_sample=single,point_delays_us=[wait]*count)
                            output=(ctypes.c_ushort*128)()
                            self.assertEqual(lib.parse(ctypes.create_string_buffer(packet),output),1)
                            self.assertEqual(list(output[:2]),[count,5])
                            self.assertEqual(list(output[2:2+2*count:2]),[r['index'] for r in rows])
                            self.assertEqual(list(output[3:2+2*count:2]),[wait]*count)
                            corrupt=bytearray(packet);corrupt[25]^=1
                            self.assertEqual(lib.parse(ctypes.create_string_buffer(bytes(corrupt)),output),0)
            # A genuine previous seven-point packet, not fabricated padding.
            import json
            fixture=root/'Python/tests/fixtures/legacy_seven_point_packets.json'
            previous=json.loads(fixture.read_text(encoding='utf-8'))
            self.assertEqual(previous['point_count'],63)
            self.assertEqual([p['single_sample'] for p in previous['packets']],[False,True])
            for old_packet in previous['packets']:
                packet=bytes.fromhex(old_packet['hex'])
                self.assertEqual(lib.parse(ctypes.create_string_buffer(packet),output),0)

if __name__ == '__main__': unittest.main(verbosity=2)
