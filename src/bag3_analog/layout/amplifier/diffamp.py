# SPDX-License-Identifier: Apache-2.0
# Copyright 2019 Blue Cheetah Analog Design Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains layout generators for differential amplifiers."""

from typing import Any, Dict, Type, Optional, List, Sequence

from pybag.enum import MinLenMode, RoundMode

from bag.typing import TrackType
from bag.util.immutable import Param
from bag.layout.template import TemplateDB
from bag.layout.routing.base import TrackID, WireArray
from bag.util.math import HalfInt
from bag.design.database import ModuleDB
from bag.design.module import Module

from xbase.layout.enum import MOSWireType, MOSPortType
from xbase.layout.mos.base import MOSBasePlaceInfo, MOSBase
from xbase.layout.mos.guardring import GuardRing

from bag3_digital.layout.stdcells.gates import InvChainCore

from ...enum import DrawTaps


class DiffAmpSelfBiased(MOSBase):
    """The core of the self biased differential amplifier
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_analog', 'diffamp_self_biased')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg_dict='Dictionary of segments',
            w_p='pmos width.',
            w_n='nmos width.',
            ridx_ntail='index for nmos row with tail transistor',
            ridx_ngm='index for nmos row with gm transistors',
            ridx_pgm='index for pmos row with gm transistors',
            ridx_ptail='index for pmos row with head transistors',
            show_pins='True to show pins',
            flip_tile='True to flip all tiles',
            draw_taps='LEFT or RIGHT or BOTH or NONE',
            sig_locs='Signal locations for top horizontal metal layer pins',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_p=0,
            w_n=0,
            ridx_ntail=0,
            ridx_ngm=1,
            ridx_pgm=-2,
            ridx_ptail=-1,
            show_pins=False,
            flip_tile=False,
            draw_taps='NONE',
            sig_locs={},
        )

    def draw_layout(self) -> None:
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo, flip_tile=self.params['flip_tile'])

        _pinfo = self.get_tile_pinfo(tile_idx=1)

        seg_dict: Dict[str, int] = self.params['seg_dict']
        w_p: int = self.params['w_p']
        w_n: int = self.params['w_n']
        ridx_ntail: int = self.params['ridx_ntail']
        ridx_ngm: int = self.params['ridx_ngm']
        ridx_pgm: int = self.params['ridx_pgm']
        ridx_ptail: int = self.params['ridx_ptail']
        draw_taps: DrawTaps = DrawTaps[self.params['draw_taps']]
        sig_locs: Dict[str, TrackType] = self.params['sig_locs']

        for key, val in seg_dict.items():
            if val % 2:
                raise ValueError(f'This generator does not support odd number of segments '
                                 f'{key} = {val}')

        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1

        seg_tailn = seg_dict['tail_n']
        seg_gmn = seg_dict['gm_n']
        seg_gmp = seg_dict['gm_p']
        seg_tailp = seg_dict['tail_p']

        # taps
        sub_sep = self.sub_sep_col
        sup_info = self.get_supply_column_info(xm_layer)
        num_taps = 0
        tap_offset = 0
        tap_left = tap_right = False
        if draw_taps in DrawTaps.RIGHT | DrawTaps.BOTH:
            num_taps += 1
            tap_right = True
        if draw_taps in DrawTaps.LEFT | DrawTaps.BOTH:
            num_taps += 1
            tap_offset += sup_info.ncol + sub_sep // 2
            tap_left = True

        # set total number of columns
        # Total width can be limited by either transistor size or by vertical metal size
        num, locs = self.tr_manager.place_wires(vm_layer, ['sig_in', 'sig', 'sig', 'sig_in'])
        sig_vm_w = self.tr_manager.get_width(vm_layer, 'sig_in')
        vm_coord = self.grid.get_wire_bounds(vm_layer, locs[-1], width=sig_vm_w)[1]
        vm_col = _pinfo.coord_to_col(vm_coord, RoundMode.GREATER_EQ)
        vm_col += vm_col & 1

        seg_max = max(seg_tailn, 2 * seg_gmn, 2 * seg_gmp, seg_tailp, vm_col)
        seg_tot = seg_max + (sup_info.ncol + sub_sep // 2) * num_taps
        self.set_mos_size(seg_tot)
        seg_tot2 = seg_max // 2 + tap_offset

        # --- Placement --- #
        if (seg_tailn // 2) % 2 == 1:
            sup_term_n, share_term_n = MOSPortType.S, MOSPortType.D
            g_on_s_n = True
        else:
            sup_term_n, share_term_n = MOSPortType.D, MOSPortType.S
            g_on_s_n = False

        # nmos tail
        ntail_inst = self.add_mos(ridx_ntail, seg_tot2 - seg_tailn // 2, seg_tailn, w=w_n,
                                  g_on_s=g_on_s_n)
        _row_info = _pinfo.get_row_place_info(ridx_ntail).row_info
        w_tailn, th_tailn = _row_info.width, _row_info.threshold

        # nmos gm cells
        ngm_left_inst = self.add_mos(ridx_ngm, seg_tot2, seg_gmn, w=w_n, flip_lr=True)
        ngm_right_inst = self.add_mos(ridx_ngm, seg_tot2, seg_gmn, w=w_n)
        _row_info = _pinfo.get_row_place_info(ridx_ngm).row_info
        w_gmn, th_gmn = _row_info.width, _row_info.threshold

        # pmos gm cells
        pgm_left_inst = self.add_mos(ridx_pgm, seg_tot2, seg_gmp, w=w_p, flip_lr=True)
        pgm_right_inst = self.add_mos(ridx_pgm, seg_tot2, seg_gmp, w=w_p)
        _row_info = _pinfo.get_row_place_info(ridx_pgm).row_info
        w_gmp, th_gmp = _row_info.width, _row_info.threshold

        if (seg_tailp // 2) % 2 == 1:
            sup_term_p, share_term_p = MOSPortType.S, MOSPortType.D
            g_on_s_p = True
        else:
            sup_term_p, share_term_p = MOSPortType.D, MOSPortType.S
            g_on_s_p = False

        # pmos tail
        ptail_inst = self.add_mos(ridx_ptail, seg_tot2 - seg_tailp // 2, seg_tailp, w=w_p,
                                  g_on_s=g_on_s_p)
        _row_info = _pinfo.get_row_place_info(ridx_ptail).row_info
        w_tailp, th_tailp = _row_info.width, _row_info.threshold

        # add taps
        lay_range = range(self.conn_layer, xm_layer + 1)
        vdd_table: Dict[int, List[WireArray]] = {lay: [] for lay in lay_range}
        vss_table: Dict[int, List[WireArray]] = {lay: [] for lay in lay_range}
        if tap_left:
            self.add_supply_column(sup_info, 0, vdd_table, vss_table)
        if tap_right:
            self.add_supply_column(sup_info, seg_tot, vdd_table, vss_table, flip_lr=True)

        # --- Routing --- #
        # 1. source terminals of tail transistors go to supplies
        vdd_tid = self.get_track_id(ridx_ptail, MOSWireType.DS, wire_name='sup')
        vdd = self.connect_to_tracks(ptail_inst[sup_term_p], vdd_tid)
        vdd_table[hm_layer].append(vdd)
        vdd_hm = self.connect_wires(vdd_table[hm_layer])
        self.add_pin('VDD_conn', vdd_table[self.conn_layer], hide=True)
        self.add_pin('VDD_hm', vdd_hm, hide=True)
        self.add_pin('VDD_vm', vdd_table[vm_layer], hide=True)
        self.add_pin('VDD', self.connect_wires(vdd_table[xm_layer]))

        vss_tid = self.get_track_id(ridx_ntail, MOSWireType.DS, wire_name='sup')
        vss = self.connect_to_tracks(ntail_inst[sup_term_n], vss_tid)
        vss_table[hm_layer].append(vss)
        vss_hm = self.connect_wires(vss_table[hm_layer])
        self.add_pin('VSS_conn', vss_table[self.conn_layer], hide=True)
        self.add_pin('VSS_hm', vss_hm, hide=True)
        self.add_pin('VSS_vm', vss_table[vm_layer], hide=True)
        self.add_pin('VSS', self.connect_wires(vss_table[xm_layer]))

        # 2. s terminals of gm transistors connect to drain terminals of tail
        tail_p_tid = self.get_track_id(ridx_ptail, MOSWireType.DS, wire_name='sig_in')
        self.connect_to_tracks([pgm_left_inst.s, pgm_right_inst.s, ptail_inst[share_term_p]],
                               tail_p_tid)
        tail_n_tid = self.get_track_id(ridx_ntail, MOSWireType.DS, wire_name='sig_in')
        self.connect_to_tracks([ngm_left_inst.s, ngm_right_inst.s, ntail_inst[share_term_n]],
                               tail_n_tid)

        # 3.  Connect d terminals of gm transistors to gate of tail
        gate_n_tid = self.get_track_id(ridx_ntail, MOSWireType.G, wire_name='sig_in')
        self.connect_to_tracks(ngm_left_inst.d, gate_n_tid)
        self.connect_to_tracks(ntail_inst.g, gate_n_tid)

        gate_p_tid = self.get_track_id(ridx_ptail, MOSWireType.G, wire_name='sig_in')
        self.connect_to_tracks(pgm_left_inst.d, gate_p_tid)
        self.connect_to_tracks(ptail_inst.g, gate_p_tid)

        # 4. get drain connections of gm transistors on horizontal metal
        clk_hm_w = self.tr_manager.get_width(hm_layer, 'sig')
        drain_n_idx0 = self.get_track_index(ridx_ngm, MOSWireType.DS, wire_name='sig', wire_idx=0)
        drain_n_idx1 = self.get_track_index(ridx_ngm, MOSWireType.DS, wire_name='sig', wire_idx=1)
        drain_left_n, drain_right_n = self.connect_differential_tracks(ngm_left_inst.d,
                                                                       ngm_right_inst.d, hm_layer,
                                                                       drain_n_idx0, drain_n_idx1,
                                                                       width=clk_hm_w)

        drain_p_idx0 = self.get_track_index(ridx_pgm, MOSWireType.DS, wire_name='sig', wire_idx=0)
        drain_p_idx1 = self.get_track_index(ridx_pgm, MOSWireType.DS, wire_name='sig', wire_idx=1)
        drain_left_p, drain_right_p = self.connect_differential_tracks(pgm_left_inst.d,
                                                                       pgm_right_inst.d, hm_layer,
                                                                       drain_p_idx0, drain_p_idx1,
                                                                       width=clk_hm_w)

        # 5. get gate connections of gm transistors on horizontal metal
        sig_hm_w = self.tr_manager.get_width(hm_layer, 'sig_in')
        gate_n_idx0 = self.get_track_index(ridx_ngm, MOSWireType.G, wire_name='sig_in', wire_idx=0)
        gate_n_idx1 = self.get_track_index(ridx_ngm, MOSWireType.G, wire_name='sig_in', wire_idx=1)
        gate_left_n, gate_right_n = self.connect_differential_tracks(ngm_left_inst.g,
                                                                     ngm_right_inst.g, hm_layer,
                                                                     gate_n_idx0, gate_n_idx1,
                                                                     width=sig_hm_w)

        gate_p_idx0 = self.get_track_index(ridx_pgm, MOSWireType.G, wire_name='sig_in', wire_idx=0)
        gate_p_idx1 = self.get_track_index(ridx_pgm, MOSWireType.G, wire_name='sig_in', wire_idx=1)
        gate_left_p, gate_right_p = self.connect_differential_tracks(pgm_left_inst.g,
                                                                     pgm_right_inst.g, hm_layer,
                                                                     gate_p_idx0, gate_p_idx1,
                                                                     width=sig_hm_w)

        # 6. vertical metal connections
        # a) find available tracks
        vm_mid = self.grid.coord_to_track(vm_layer, drain_left_n.middle, mode=RoundMode.NEAREST)
        try:
            loc_mid = (locs[0] + locs[-1]) / 2
        except ValueError:
            loc_mid = (locs[0] + locs[-1]) // 2 + HalfInt(1)
        vm_offset = vm_mid - loc_mid
        inp_idx = locs[0] + vm_offset
        inn_idx = locs[-1] + vm_offset
        tail_g_idx = locs[1] + vm_offset
        out_idx = locs[-2] + vm_offset

        # b) connect
        inp, inn = self.connect_differential_tracks([gate_left_p, gate_left_n],
                                                    [gate_right_p, gate_right_n], vm_layer,
                                                    inp_idx, inn_idx, width=sig_vm_w)

        clk_vm_w = self.tr_manager.get_width(vm_layer, 'sig')
        tail_g, out = self.connect_differential_tracks([drain_left_p, drain_left_n],
                                                       [drain_right_p, drain_right_n], vm_layer,
                                                       tail_g_idx, out_idx, width=clk_vm_w)

        # 7. get all pins on top horizontal metal
        out_idx = sig_locs.get('out', self.grid.coord_to_track(xm_layer, out.middle,
                                                               mode=RoundMode.NEAREST))
        out = self.connect_to_tracks(out, TrackID(xm_layer, out_idx,
                                     width=self.tr_manager.get_width(xm_layer, 'sig')),
                                     min_len_mode=MinLenMode.UPPER)
        inp_idx = sig_locs.get('inp', self.tr_manager.get_next_track(xm_layer, out_idx, 'sig',
                                                                     'sig_in', up=True))
        inn_idx = sig_locs.get('inn', self.tr_manager.get_next_track(xm_layer, out_idx, 'sig',
                                                                     'sig_in', up=False))

        inp, inn = self.connect_differential_tracks(inp, inn, xm_layer, inp_idx, inn_idx,
                                                    width=self.tr_manager.get_width(xm_layer,
                                                                                    'sig_in'))

        # pins
        self.add_pin('v_inp', inp, show=self.show_pins)
        self.add_pin('v_inn', inn, show=self.show_pins)
        self.add_pin('v_out', out, show=self.show_pins)

        # set properties
        self.sch_params = dict(
            lch=_pinfo.lch,
            seg_dict=dict(
                gm_n=seg_gmn,
                gm_p=seg_gmp,
                tail_n=seg_tailn,
                tail_p=seg_tailp,
            ),
            w_dict=dict(
                gm_n=w_gmn,
                gm_p=w_gmp,
                tail_n=w_tailn,
                tail_p=w_tailp,
            ),
            th_dict=dict(
                gm_n=th_gmn,
                gm_p=th_gmp,
                tail_n=th_tailn,
                tail_p=th_tailp,
            ),
        )


class DiffAmpSelfBiasedBuffer(MOSBase):
    """self biased differential amplifier followed bu buffer
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_analog', 'diffamp_self_biased_buffer')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg_dict='Dictionary of segments for diffamp core',
            segp_list='List of pmos segments per stage of InvChain',
            segn_list='List of nmos segments per stage of InvChain',
            w_p='pmos width.',
            w_n='nmos width.',
            ridx_ntail='index for nmos row with tail transistor',
            ridx_ngm='index for nmos row with gm transistors',
            ridx_pgm='index for pmos row with gm transistors',
            ridx_ptail='index for pmos row with head transistors',
            show_pins='True to show pins',
            flip_tile='True to flip all tiles',
            draw_taps='LEFT or RIGHT or BOTH or NONE',
            sig_locs='Signal locations for top horizontal metal layer pins',
            export_mid='True to export mid',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_p=0,
            w_n=0,
            ridx_ntail=0,
            ridx_ngm=1,
            ridx_pgm=-2,
            ridx_ptail=-1,
            show_pins=False,
            flip_tile=False,
            draw_taps='NONE',
            sig_locs={},
            export_mid=False,
        )

    def draw_layout(self) -> None:
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        draw_taps: DrawTaps = DrawTaps[self.params['draw_taps']]
        export_mid: bool = self.params['export_mid']
        ridx_ngm: int = self.params['ridx_ngm']
        ridx_pgm: int = self.params['ridx_pgm']

        # create masters
        diffamp_params = self.params.copy(remove=['draw_taps'], append=dict(show_pins=False))
        diffamp_master = self.new_template(DiffAmpSelfBiased, params=diffamp_params)
        diffamp_ncol = diffamp_master.num_cols

        segp_list: Sequence[int] = self.params['segp_list']
        segn_list: Sequence[int] = self.params['segn_list']
        if len(segp_list) % 2 == 1 or len(segn_list) % 2 == 1:
            raise ValueError('Buffer should have even length to preserve polarity.')
        buf_params = self.params.copy(append=dict(is_guarded=True, show_pins=False,
                                                  ridx_p=ridx_pgm, ridx_n=ridx_ngm,
                                                  vertical_sup=True))
        buf_master = self.new_template(InvChainCore, params=buf_params)
        buf_ncol = buf_master.num_cols

        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1

        sep = max(self.min_sep_col, self.get_hm_sp_le_sep_col())
        # taps
        sub_sep = self.sub_sep_col
        sup_info = self.get_supply_column_info(xm_layer)
        num_taps = 0
        tap_offset = 0
        tap_left = tap_right = False
        if draw_taps in DrawTaps.RIGHT | DrawTaps.BOTH:
            num_taps += 1
            tap_right = True
        if draw_taps in DrawTaps.LEFT | DrawTaps.BOTH:
            num_taps += 1
            tap_offset += sup_info.ncol + sub_sep // 2
            tap_left = True

        # set total number of columns
        seg_tot = diffamp_ncol + sep + buf_ncol + (sup_info.ncol + sub_sep // 2) * num_taps
        self.set_mos_size(seg_tot)

        # --- Placement --- #
        cur_col = tap_offset
        diffamp_inst = self.add_tile(diffamp_master, 0, cur_col)
        cur_col += diffamp_ncol + sep

        buf_inst = self.add_tile(buf_master, 0, cur_col)

        # add taps
        lay_range = range(self.conn_layer, xm_layer + 1)
        vdd_table: Dict[int, List[WireArray]] = {lay: [] for lay in lay_range}
        vss_table: Dict[int, List[WireArray]] = {lay: [] for lay in lay_range}
        if tap_left:
            self.add_supply_column(sup_info, 0, vdd_table, vss_table)
        if tap_right:
            self.add_supply_column(sup_info, seg_tot, vdd_table, vss_table, flip_lr=True)

        # --- Routing --- #
        # 1. supplies
        vdd_table[hm_layer].append(diffamp_inst.get_pin('VDD_hm'))
        self.connect_to_track_wires(buf_inst.get_all_port_pins('VDD'), vdd_table[hm_layer])
        self.add_pin('VDD_conn', vdd_table[self.conn_layer], hide=True)
        self.add_pin('VDD_hm', vdd_table[hm_layer], hide=True)
        self.add_pin('VDD_vm', vdd_table[vm_layer], hide=True)
        self.add_pin('VDD', self.connect_wires(vdd_table[xm_layer]))

        vss_table[hm_layer].append(diffamp_inst.get_pin('VSS_hm'))
        self.connect_to_track_wires(buf_inst.get_all_port_pins('VSS'), vss_table[hm_layer])
        self.add_pin('VSS_conn', vss_table[self.conn_layer], hide=True)
        self.add_pin('VSS_hm', vss_table[hm_layer], hide=True)
        self.add_pin('VSS_vm', vss_table[vm_layer], hide=True)
        self.add_pin('VSS', self.connect_wires(vss_table[xm_layer]))

        # 2. export inp, inn
        self.add_pin('v_inp', self.extend_wires(diffamp_inst.get_pin('v_inp'), lower=0))
        self.add_pin('v_inn', self.extend_wires(diffamp_inst.get_pin('v_inn'), lower=0))

        # 3. connect diffamp output to buffer input
        mid_vm = diffamp_inst.get_pin('v_out')
        mid = self.connect_to_track_wires(buf_inst.get_pin('in'), mid_vm)
        self.add_pin('v_mid', mid, hide=not export_mid)

        # 4. get final output on top horizontal metal
        out = self.connect_to_tracks(buf_inst.get_pin('out'), mid_vm.track_id,
                                     track_upper=self.bound_box.w)
        self.add_pin('v_out', out)

        # set properties
        self.sch_params = dict(
            diffamp_params=diffamp_master.sch_params,
            buf_params=buf_master.sch_params,
            export_mid=export_mid,
        )


class DiffAmpSelfBiasedBufferGuardRing(GuardRing):
    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        GuardRing.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        ans = DiffAmpSelfBiasedBuffer.get_params_info()
        ans.update(
            pmos_gr='pmos guard ring tile name.',
            nmos_gr='nmos guard ring tile name.',
            edge_ncol='Number of columns on guard ring edge.  Use 0 for default.',
        )
        return ans

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        ans = DiffAmpSelfBiasedBuffer.get_default_param_values()
        ans.update(
            pmos_gr='pgr',
            nmos_gr='ngr',
            edge_ncol=0,
        )
        return ans

    def get_layout_basename(self) -> str:
        return self.__class__.__name__

    def draw_layout(self) -> None:
        params = self.params
        pmos_gr: str = params['pmos_gr']
        nmos_gr: str = params['nmos_gr']
        edge_ncol: int = params['edge_ncol']

        core_params = params.copy(remove=['pmos_gr', 'nmos_gr', 'edge_ncol'])
        master = self.new_template(DiffAmpSelfBiasedBuffer, params=core_params)

        sub_sep = master.sub_sep_col
        sep_ncol_left = sep_ncol_right = sub_sep
        draw_taps: DrawTaps = DrawTaps[params['draw_taps']]
        if draw_taps in DrawTaps.RIGHT | DrawTaps.BOTH:
            sep_ncol_right = sub_sep // 2
        if draw_taps in DrawTaps.LEFT | DrawTaps.BOTH:
            sep_ncol_left = sub_sep // 2
        sep_ncol = (sep_ncol_left, sep_ncol_right)

        inst, sup_list = self.draw_guard_ring(master, pmos_gr, nmos_gr, sep_ncol, edge_ncol)
        vdd_hm_list, vss_hm_list = [], []
        for (vss_list, vdd_list) in sup_list:
            vss_hm_list.extend(vss_list)
            vdd_hm_list.extend(vdd_list)

        self.connect_to_track_wires(vss_hm_list, inst.get_all_port_pins('VSS_vm'))
        self.connect_to_track_wires(vdd_hm_list, inst.get_all_port_pins('VDD_vm'))
