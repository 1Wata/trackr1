from evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_path = ''
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_path = ''
    # settings.network_path = '/data1/lihaobo/tracking/dataset_interface/networks/'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.otb_lang_path = '/data1/lihaobo/tracking/data/OTB_lang'
    settings.tnl2k_path = '/data1/lihaobo/tracking/data/TNL2K_CVPR2021'
    settings.lasot_path = '/data1/lihaobo/tracking/data/lasot'
    settings.result_plot_path = '/data1/lihaobo/tracking/dataset_interface/result_plots/'
    settings.results_path = '/data1/lihaobo/tracking/dataset_interface/tracking_results/'    # Where to store tracking results
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot_path = ''
    settings.youtubevos_dir = ''

    return settings

