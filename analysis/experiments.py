"""Analysis settings used for each simulation campaign of this project.

M is the ACF summation window for the integrated autocorrelation time, chosen per
sampling interval; the xlim values only set the lag range of the ACF plots. Running
this file re-analyses every experiment, which needs the raw LAMMPS output in
simulations/ (not tracked in git).
"""

from lammps_analysis import analyze_experiment


def analysis_300K():
    # tsamp 10
    analyze_experiment("t300_ns100000_ne50000_nr1000000_ts10_td1000", M=30, xlim_acf_halves=200, xlim_acf=500)  # at L=180 messy
    analyze_experiment("t300_ns100000_ne50000_nr1500000_ts10_td1000", M=30, xlim_acf_halves=200, xlim_acf=500)  # at L=150 messy
    # analyze_experiment("t300_ns100000_ne100000_nr1500000_ts10_td1000", M=30, xlim_acf_halves=200, xlim_acf=1000)  # at L=140 messy
    analyze_experiment("t300_ns100000_ne1000000_nr1500000_ts10_td1000", M=30, xlim_acf_halves=200, xlim_acf=500)  # at L=140 messy

    # tsamp 20
    analyze_experiment("t300_ns100000_ne50000_nr1000000_ts20_td1000", M=15, xlim_acf_halves=200, xlim_acf=500)  # at L=180 messy
    analyze_experiment("t300_ns100000_ne50000_nr1500000_ts20_td1000", M=15, xlim_acf_halves=200, xlim_acf=500)  # at L=150 messy

    # tsamp 50
    analyze_experiment("t300_ns100000_ne100000_nr1500000_ts50_td1000", M=100, xlim_acf_halves=200, xlim_acf=500)  # at L=150 messy


def analysis_700K():
    # tsamp 10
    analyze_experiment("t700_ns10000_ne5000_nr1000000_ts10_td1000", M=90, xlim_acf_halves=200, xlim_acf=500)

    # tsamp 20
    analyze_experiment("t700_ns10000_ne10000_nr1000000_ts20_td1000", M=45, xlim_acf_halves=100, xlim_acf=500)  # at L=160,180,190 messy
    analyze_experiment("t700_ns10000_ne50000_nr1000000_ts20_td1000", M=45, xlim_acf_halves=100, xlim_acf=500)
    analyze_experiment("t700_ns100000_ne10000_nr1000000_ts20_td1000", M=45, xlim_acf_halves=100, xlim_acf=500)
    analyze_experiment("t700_ns100000_ne50000_nr1000000_ts20_td1000", M=45, xlim_acf_halves=100, xlim_acf=500)

    # tsamp 50
    analyze_experiment("t700_ns100000_ne50000_nr1000000_ts50_td1000", M=70, xlim_acf_halves=300, xlim_acf=500)

    # tsamp 100
    analyze_experiment("t700_ns10000_ne5000_nr100000_ts100_td1000", M=150, xlim_acf_halves=500, xlim_acf=500)


if __name__ == "__main__":
    analysis_300K()
    analysis_700K()
