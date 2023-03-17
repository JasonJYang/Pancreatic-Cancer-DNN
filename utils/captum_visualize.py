import os
import pandas as pd
import numpy as np

from captum.attr import IntegratedGradients, DeepLift, GradientShap, NoiseTunnel, FeatureAblation

def attributions_visulaize(model, X_test, X_train, save_dir, feature_list):
    ig = IntegratedGradients(model)
    ig_nt = NoiseTunnel(ig)
    dl = DeepLift(model)
    gs = GradientShap(model)
    fa = FeatureAblation(model)

    ig_attr_test = ig.attribute(X_test, n_steps=50)
    ig_nt_attr_test = ig_nt.attribute(X_test)
    dl_attr_test = dl.attribute(X_test)
    gs_attr_test = gs.attribute(X_test, X_train)
    fa_attr_test = fa.attribute(X_test)

    ig_attr_test_sum = ig_attr_test.cpu().detach().numpy().sum(0)
    ig_attr_test_norm_sum = ig_attr_test_sum / np.linalg.norm(ig_attr_test_sum, ord=1)

    ig_nt_attr_test_sum = ig_nt_attr_test.cpu().detach().numpy().sum(0)
    ig_nt_attr_test_norm_sum = ig_nt_attr_test_sum / np.linalg.norm(ig_nt_attr_test_sum, ord=1)
    

    dl_attr_test_sum = dl_attr_test.cpu().detach().numpy().sum(0)
    dl_attr_test_norm_sum = dl_attr_test_sum / np.linalg.norm(dl_attr_test_sum, ord=1)

    gs_attr_test_sum = gs_attr_test.cpu().detach().numpy().sum(0)
    gs_attr_test_norm_sum = gs_attr_test_sum / np.linalg.norm(gs_attr_test_sum, ord=1)

    fa_attr_test_sum = fa_attr_test.cpu().detach().numpy().sum(0)
    fa_attr_test_norm_sum = fa_attr_test_sum / np.linalg.norm(fa_attr_test_sum, ord=1)

    attribution_df = pd.DataFrame({'feature': feature_list,
                                   'IntegratedGradients': list(ig_attr_test_norm_sum),
                                   'NoiseTunnel': list(ig_nt_attr_test_norm_sum),
                                   'DeepLift': list(dl_attr_test_norm_sum),
                                   'GradientShap': list(gs_attr_test_norm_sum),
                                   'FeatureAblation': list(fa_attr_test_norm_sum)})
    attribution_df.to_csv(save_dir, index=False, encoding='utf_8_sig')