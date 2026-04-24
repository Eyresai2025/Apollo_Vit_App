import math

def area_defect_generic(t_pix, d_pix, total_area_mm2):
    return (total_area_mm2 * d_pix) / max(t_pix, 1)

def cor_generic(iwid, ilen, dwid, dlen, surface_height_mm, surface_width_mm):
    rdlen = (surface_height_mm * dlen) / max(ilen, 1)
    rdwid = (surface_width_mm * dwid) / max(iwid, 1)
    return rdlen, rdwid

def area_defect_sw(t_pix, d_pix, areaOfSidewall):
    return area_defect_generic(t_pix, d_pix, areaOfSidewall)

def area_defect_tread(t_pix, d_pix, areaOfTread):
    return area_defect_generic(t_pix, d_pix, areaOfTread)

def area_defect_bead(t_pix, d_pix, areaOfBead):
    return area_defect_generic(t_pix, d_pix, areaOfBead)

def area_defect_innerwall(t_pix, d_pix, areaOfInnerwall):
    return area_defect_generic(t_pix, d_pix, areaOfInnerwall)

def cor_sw(iwid, ilen, dwid, dlen, sidewallHeight, sidewallWidth):
    return cor_generic(iwid, ilen, dwid, dlen, sidewallHeight, sidewallWidth)

def cor_tread(iwid, ilen, dwid, dlen, treadHeight, treadWidth):
    return cor_generic(iwid, ilen, dwid, dlen, treadHeight, treadWidth)

def cor_bead(iwid, ilen, dwid, dlen, beadHeight, beadWidth):
    return cor_generic(iwid, ilen, dwid, dlen, beadHeight, beadWidth)

def cor_innerwall(iwid, ilen, dwid, dlen, innerwallHeight, innerwallWidth):
    return cor_generic(iwid, ilen, dwid, dlen, innerwallHeight, innerwallWidth)