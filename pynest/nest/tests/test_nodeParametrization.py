# -*- coding: utf-8 -*-
#
# test_nodeParametrization.py
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.

"""
Node Parametrization tests
"""

import nest
import numpy as np
import unittest


class TestNodeParametrization(unittest.TestCase):

    def setUp(self):
        nest.ResetKernel()

    def test_create_with_list(self):
        """Test Create with list as parameter"""
        Vm_ref = [-11., -12., -13.]
        nodes = nest.Create('iaf_psc_alpha', 3, {'V_m': Vm_ref})

        self.assertEqual(list(nest.GetStatus(nodes, 'V_m')), Vm_ref)

    def test_create_with_several_lists(self):
        """Test Create with several lists as parameters"""
        Vm_ref = [-22., -33., -44.]
        Cm_ref = 124.
        Vmin_ref = [-1., -2., -3.]

        nodes = nest.Create('iaf_psc_alpha', 3, {'V_m': Vm_ref,
                                                 'C_m': Cm_ref,
                                                 'V_min': Vmin_ref})

        self.assertEqual(list(nest.GetStatus(nodes, 'V_m')), Vm_ref)
        self.assertEqual(nest.GetStatus(nodes, 'C_m'),
                         (Cm_ref, Cm_ref, Cm_ref))
        self.assertEqual(list(nest.GetStatus(nodes, 'V_min')), Vmin_ref)

    def test_create_with_spike_generator(self):
        """Test Create with list that should not be split"""
        spike_times = [10., 20., 30.]
        sg = nest.Create('spike_generator', 2, {'spike_times': spike_times})

        st = nest.GetStatus(sg, 'spike_times')

        self.assertEqual(list(st[0]), spike_times)
        self.assertEqual(list(st[1]), spike_times)

    def test_create_with_numpy(self):
        """Test Create with numpy array as parameter"""
        Vm_ref = [-80., -90., -100.]
        nodes = nest.Create('iaf_psc_alpha', 3, {'V_m': np.array(Vm_ref)})

        self.assertEqual(list(nest.GetStatus(nodes, 'V_m')), Vm_ref)

    def test_create_uniform(self):
        """Test Create with random.uniform as parameter"""
        min_val = -75.
        max_val = -55.
        nodes = nest.Create('iaf_psc_alpha', 3,
                            {'V_m': nest.random.uniform(
                                min=min_val, max=max_val)})
        for vm in nodes.get('V_m'):
            self.assertGreaterEqual(vm, min_val)
            self.assertLessEqual(vm, max_val)

    def test_create_normal(self):
        """Test Create with random.normal as parameter"""
        nodes = nest.Create('iaf_psc_alpha', 3,
                            {'V_m': nest.random.normal(
                                loc=10.0, scale=5.0, min=0.5)})
        for vm in nodes.get('V_m'):
            self.assertGreaterEqual(vm, 0.5)

    def test_create_exponential(self):
        """Test Create with random.exonential as parameter"""
        nodes = nest.Create('iaf_psc_alpha', 3,
                            {'V_m': nest.random.exponential(scale=1.0)})
        for vm in nodes.get('V_m'):
            self.assertGreaterEqual(vm, 0.)

    def test_create_lognormal(self):
        """Test Create with random.lognormal as parameter"""
        nodes = nest.Create('iaf_psc_alpha', 3,
                            {'V_m': nest.random.lognormal(
                                mean=10., sigma=20.)})
        for vm in nodes.get('V_m'):
            self.assertGreaterEqual(vm, 0.)

    def test_create_adding(self):
        """Test Create with different parameters added"""
        nodes = nest.Create('iaf_psc_alpha', 3,
                            {'V_m': -80.0 +
                             nest.random.exponential(scale=0.1)})

        for vm in nodes.get('V_m'):
            self.assertGreaterEqual(vm, -80.0)

        nodes = nest.Create('iaf_psc_alpha', 3,
                            {'V_m': 30.0 + nest.random.uniform(-75., -55.)})

        for vm in nodes.get('V_m'):
            self.assertGreaterEqual(vm, -45.)
            self.assertLessEqual(vm, -25.)

    def test_SetStatus_with_dict(self):
        """Test SetStatus with dict"""
        nodes = nest.Create('iaf_psc_alpha', 3)
        Vm_ref = (-60., -60., -60.)
        nest.SetStatus(nodes, {'V_m': -60.})

        self.assertEqual(nest.GetStatus(nodes, 'V_m'), Vm_ref)

    def test_SetStatus_with_dict_several(self):
        """Test SetStatus with multivalue dict"""
        nodes = nest.Create('iaf_psc_alpha', 3)
        Vm_ref = (-27., -27., -27.)
        Cm_ref = (111., 111., 111.)
        nest.SetStatus(nodes, {'V_m': -27., 'C_m': 111.})

        self.assertEqual(nest.GetStatus(nodes, 'V_m'), Vm_ref)
        self.assertEqual(nest.GetStatus(nodes, 'C_m'), Cm_ref)

    def test_SetStatus_with_list_with_dicts(self):
        """Test SetStatus with list of dicts"""
        nodes = nest.Create('iaf_psc_alpha', 3)
        Vm_ref = (-70., -20., -88.)
        nest.SetStatus(nodes, [{'V_m': -70.}, {'V_m': -20.}, {'V_m': -88.}])

        self.assertEqual(nest.GetStatus(nodes, 'V_m'), Vm_ref)

    def test_SetStatus_with_dict_with_single_list(self):
        """Test SetStatus with dict with list"""

        nodes = nest.Create('iaf_psc_alpha', 3)
        Vm_ref = [-30., -40., -50.]
        nest.SetStatus(nodes, {'V_m': Vm_ref})

        self.assertEqual(list(nest.GetStatus(nodes, 'V_m')), Vm_ref)

    def test_SetStatus_with_dict_with_lists(self):
        """Test SetStatus with dict with lists"""
        nodes = nest.Create('iaf_psc_alpha', 3)
        Vm_ref = [-11., -12., -13.]
        Cm_ref = 177.
        tau_minus_ref = [22., 24., 26.]
        nest.SetStatus(nodes, {'V_m': Vm_ref,
                               'C_m': Cm_ref,
                               'tau_minus': tau_minus_ref})

        self.assertEqual(list(nest.GetStatus(nodes, 'V_m')), Vm_ref)
        self.assertEqual(nest.GetStatus(nodes, 'C_m'),
                         (Cm_ref, Cm_ref, Cm_ref))
        self.assertEqual(list(nest.GetStatus(nodes, 'tau_minus')),
                         tau_minus_ref)

    def test_SetStatus_with_dict_with_single_element_lists(self):
        """Test SetStatus with dict with single element lists"""
        node = nest.Create('iaf_psc_alpha')
        Vm_ref = (-13.,)
        Cm_ref = (222.,)
        nest.SetStatus(node, {'V_m': [-13.], 'C_m': [222.]})

        self.assertEqual(nest.GetStatus(node, 'V_m'), Vm_ref)
        self.assertEqual(nest.GetStatus(node, 'C_m'), Cm_ref)

    def test_SetStatus_with_dict_with_string(self):
        """Test SetStatus with dict with bool"""
        nodes = nest.Create('spike_detector', 3)
        withport_ref = (True, True, True)
        nest.SetStatus(nodes, {'withport': True})

        self.assertEqual(nest.GetStatus(nodes, 'withport'), withport_ref)

    def test_SetStatus_with_dict_with_list_with_strings(self):
        """Test SetStatus with dict with list of bools"""
        nodes = nest.Create('spike_detector', 3)
        withport_ref = (True, False, True)
        nest.SetStatus(nodes, {'withport': [True, False, True]})

        self.assertEqual(nest.GetStatus(nodes, 'withport'), withport_ref)

    def test_SetStatus_on_spike_generetor(self):
        """Test SetStatus with dict with list that is not to be split"""
        sg = nest.Create('spike_generator')
        nest.SetStatus(sg, {'spike_times': [1., 2., 3.]})

        self.assertEqual(list(nest.GetStatus(sg, 'spike_times')[0]),
                         [1., 2., 3.])

    def test_SetStatus_with_dict_with_numpy(self):
        """Test SetStatus with dict with numpy"""
        nodes = nest.Create('iaf_psc_alpha', 3)

        Vm_ref = np.array([-22., -33., -44.])
        nest.SetStatus(nodes, {'V_m': Vm_ref})

        self.assertEqual(list(nest.GetStatus(nodes, 'V_m')), list(Vm_ref))

    def test_SetStatus_with_random(self):
        """Test SetStatus with dict with random.uniform"""
        nodes = nest.Create('iaf_psc_alpha', 3)
        nest.SetStatus(nodes, {'V_m': nest.random.uniform(-75., -55.)})

        for vm in nodes.get('V_m'):
            self.assertGreater(vm, -75.)
            self.assertLess(vm, -55.)

    def test_SetStatus_with_random_as_val(self):
        """Test SetStatus with val as random.uniform"""
        nodes = nest.Create('iaf_psc_alpha', 3)
        nest.SetStatus(nodes, 'V_m', nest.random.uniform(-75., -55.))

        for vm in nodes.get('V_m'):
            self.assertGreater(vm, -75.)
            self.assertLess(vm, -55.)

    def test_set_with_dict_with_single_list(self):
        """Test set with dict with list"""
        nodes = nest.Create('iaf_psc_alpha', 3)
        Vm_ref = [-30., -40., -50.]
        nodes.set({'V_m': Vm_ref})

        self.assertEqual(list(nodes.get('V_m')), Vm_ref)

    def test_set_with_dict_with_lists(self):
        """Test set with dict with lists"""
        nodes = nest.Create('iaf_psc_alpha', 3)
        Vm_ref = [-11., -12., -13.]
        Cm_ref = 177.
        tau_minus_ref = [22., 24., 26.]
        nodes.set({'V_m': Vm_ref,
                   'C_m': Cm_ref,
                   'tau_minus': tau_minus_ref})

        self.assertEqual(list(nodes.get('V_m')), Vm_ref)
        self.assertEqual(nodes.get('C_m'), (Cm_ref, Cm_ref, Cm_ref))
        self.assertEqual(list(nodes.get('tau_minus')), tau_minus_ref)

    def test_set_with_dict_with_single_element_lists(self):
        """Test set with dict with single element lists"""
        node = nest.Create('iaf_psc_alpha')
        Vm_ref = -13.
        Cm_ref = 222.
        node.set({'V_m': [Vm_ref], 'C_m': [Cm_ref]})

        self.assertEqual(node.get('V_m'), Vm_ref)
        self.assertEqual(node.get('C_m'), Cm_ref)

    def test_set_with_dict_with_list_with_strings(self):
        """Test set with dict with list with bool"""
        nodes = nest.Create('spike_detector', 3)
        withport_ref = (True, False, True)
        nodes.set({'withport': [True, False, True]})

        self.assertEqual(nodes.get('withport'), withport_ref)

    def test_set_on_spike_generetor(self):
        """Test set with dict with list that is not to be split"""
        sg = nest.Create('spike_generator')
        sg.set({'spike_times': [1., 2., 3.]})

        self.assertEqual(list(sg.get('spike_times')), [1., 2., 3.])

    def test_set_with_random(self):
        """Test set with dict with random parameter"""
        nodes = nest.Create('iaf_psc_alpha', 3)
        nodes.set({'V_m': nest.random.uniform(-75., -55.)})

        for vm in nodes.get('V_m'):
            self.assertGreater(vm, -75.)
            self.assertLess(vm, -55.)

    def test_set_with_random_as_val(self):
        """Test set with random parameter as val"""
        nodes = nest.Create('iaf_psc_alpha', 3)
        nodes.set('V_m', nest.random.uniform(-75., -55.))

        for vm in nodes.get('V_m'):
            self.assertGreater(vm, -75.)
            self.assertLess(vm, -55.)


def suite():
    suite = unittest.makeSuite(TestNodeParametrization, 'test')
    return suite


def run():
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())


if __name__ == "__main__":
    run()