inline void get_field_value_pullback(const float *y, int field_type, int i, int j, int k, int xN, int yN, int zN, const float *u_inlet, float _d_y0, float *_d_y, int *_d_field_type, int *_d_i, int *_d_j, int *_d_k, int *_d_xN, int *_d_yN, int *_d_zN) {
    bool _cond0;
    bool _cond1;
    bool _cond2;
    bool _cond3;
    bool _cond4;
    bool _cond5;
    bool _cond6;
    bool _cond7;
    bool _cond8;
    bool _cond9;
    bool _cond10;
    bool _cond11;
    int _d_sizeY = 0;
    int sizeY = yN + 2;
    int _d_sizeZ = 0;
    int sizeZ = zN + 2;
    int _d_nCell = 0;
    int nCell = xN * yN * zN;
    int _d_i_clamped = 0;
    int i_clamped = i;
    int _d_j_clamped = 0;
    int j_clamped = j;
    int _d_k_clamped = 0;
    int k_clamped = k;
    {
        _cond0 = i < 1;
        if (_cond0) {
            i_clamped = 1;
            {
                _cond1 = field_type == 0;
                if (_cond1) {
                    goto _label0;
                }
            }
            goto _label1;
        }
    }
    {
        _cond2 = i > xN;
        if (_cond2) {
            i_clamped = xN;
            {
                _cond3 = field_type == 3;
                if (_cond3) {
                    goto _label2;
                }
            }
        }
    }
    {
        _cond4 = j < 1;
        if (_cond4) {
            j_clamped = 1;
            {
                _cond5 = field_type != 3;
                if (_cond5)
                    goto _label3;
            }
        }
    }
    {
        _cond6 = j > yN;
        if (_cond6) {
            j_clamped = yN;
            {
                _cond7 = field_type != 3;
                if (_cond7)
                    goto _label4;
            }
        }
    }
    {
        _cond8 = k < 1;
        if (_cond8) {
            k_clamped = 1;
            {
                _cond9 = field_type != 3;
                if (_cond9)
                    goto _label5;
            }
        }
    }
    {
        _cond10 = k > zN;
        if (_cond10) {
            k_clamped = zN;
            {
                _cond11 = field_type != 3;
                if (_cond11)
                    goto _label6;
            }
        }
    }
    int _d_pos = 0;
    const int pos = (i_clamped - 1) * yN * zN + (j_clamped - 1) * zN + (k_clamped - 1);
    _d_y[field_type * nCell + pos] += _d_y0;
    {
        _d_i_clamped += _d_pos * zN * yN;
        *_d_yN += (i_clamped - 1) * _d_pos * zN;
        *_d_zN += (i_clamped - 1) * yN * _d_pos;
        _d_j_clamped += _d_pos * zN;
        *_d_zN += (j_clamped - 1) * _d_pos;
        _d_k_clamped += _d_pos;
    }
    if (_cond10) {
        if (_cond11)
          _label6:
            ;
        {
            *_d_zN += _d_k_clamped;
            _d_k_clamped = 0;
        }
    }
    if (_cond8) {
        if (_cond9)
          _label5:
            ;
        _d_k_clamped = 0;
    }
    if (_cond6) {
        if (_cond7)
          _label4:
            ;
        {
            *_d_yN += _d_j_clamped;
            _d_j_clamped = 0;
        }
    }
    if (_cond4) {
        if (_cond5)
          _label3:
            ;
        _d_j_clamped = 0;
    }
    if (_cond2) {
        if (_cond3) {
          _label2:
            ;
        }
        {
            *_d_xN += _d_i_clamped;
            _d_i_clamped = 0;
        }
    }
    if (_cond0) {
      _label1:
        ;
        if (_cond1) {
          _label0:
            ;
        }
        _d_i_clamped = 0;
    }
    *_d_k += _d_k_clamped;
    *_d_j += _d_j_clamped;
    *_d_i += _d_i_clamped;
    {
        *_d_xN += _d_nCell * zN * yN;
        *_d_yN += xN * _d_nCell * zN;
        *_d_zN += xN * yN * _d_nCell;
    }
    *_d_zN += _d_sizeZ;
    *_d_yN += _d_sizeY;
}
void compute_residual_component_grad_1(float Re, const float *y, int i, int j, int k, int eq_type, int xN, int yN, int zN, float dx, float dy, float dz, const float *u_inlet, float *_d_y) {
    float _d_Re = 0.F;
    int _d_i = 0;
    int _d_j = 0;
    int _d_k = 0;
    int _d_eq_type = 0;
    int _d_xN = 0;
    int _d_yN = 0;
    int _d_zN = 0;
    float _d_dx = 0.F;
    float _d_dy = 0.F;
    float _d_dz = 0.F;
    bool _cond0;
    float _d_u_ip1 = 0.F;
    float u_ip1 = 0.F;
    float _d_u_im1 = 0.F;
    float u_im1 = 0.F;
    float _d_u_jp1 = 0.F;
    float u_jp1 = 0.F;
    float _d_u_jm1 = 0.F;
    float u_jm1 = 0.F;
    float _d_u_kp1 = 0.F;
    float u_kp1 = 0.F;
    float _d_u_km1 = 0.F;
    float u_km1 = 0.F;
    float _d_u_ijk = 0.F;
    float u_ijk = 0.F;
    float _d_v_jp1 = 0.F;
    float v_jp1 = 0.F;
    float _d_v_jm1 = 0.F;
    float v_jm1 = 0.F;
    float _d_w_kp1 = 0.F;
    float w_kp1 = 0.F;
    float _d_w_km1 = 0.F;
    float w_km1 = 0.F;
    float _d_p_ip1 = 0.F;
    float p_ip1 = 0.F;
    float _d_p_ijk = 0.F;
    float p_ijk = 0.F;
    float _d_conv_x = 0.F;
    float conv_x = 0.F;
    float _d_conv_y = 0.F;
    float conv_y = 0.F;
    float _d_conv_z = 0.F;
    float conv_z = 0.F;
    float _d_pres = 0.F;
    float pres = 0.F;
    float _d_diff = 0.F;
    float diff = 0.F;
    bool _cond1;
    float _d_v_ip1 = 0.F;
    float v_ip1 = 0.F;
    float _d_v_im1 = 0.F;
    float v_im1 = 0.F;
    float _d_v_jp10 = 0.F;
    float v_jp10 = 0.F;
    float _d_v_jm10 = 0.F;
    float v_jm10 = 0.F;
    float _d_v_kp1 = 0.F;
    float v_kp1 = 0.F;
    float _d_v_km1 = 0.F;
    float v_km1 = 0.F;
    float _d_v_ijk = 0.F;
    float v_ijk = 0.F;
    float _d_u_ip10 = 0.F;
    float u_ip10 = 0.F;
    float _d_u_im10 = 0.F;
    float u_im10 = 0.F;
    float _d_w_kp10 = 0.F;
    float w_kp10 = 0.F;
    float _d_w_km10 = 0.F;
    float w_km10 = 0.F;
    float _d_p_jp1 = 0.F;
    float p_jp1 = 0.F;
    float _d_p_ijk0 = 0.F;
    float p_ijk0 = 0.F;
    float _d_conv_x0 = 0.F;
    float conv_x0 = 0.F;
    float _d_conv_y0 = 0.F;
    float conv_y0 = 0.F;
    float _d_conv_z0 = 0.F;
    float conv_z0 = 0.F;
    float _d_pres0 = 0.F;
    float pres0 = 0.F;
    float _d_diff0 = 0.F;
    float diff0 = 0.F;
    bool _cond2;
    float _d_w_ip1 = 0.F;
    float w_ip1 = 0.F;
    float _d_w_im1 = 0.F;
    float w_im1 = 0.F;
    float _d_w_jp1 = 0.F;
    float w_jp1 = 0.F;
    float _d_w_jm1 = 0.F;
    float w_jm1 = 0.F;
    float _d_w_kp11 = 0.F;
    float w_kp11 = 0.F;
    float _d_w_km11 = 0.F;
    float w_km11 = 0.F;
    float _d_w_ijk = 0.F;
    float w_ijk = 0.F;
    float _d_u_ip11 = 0.F;
    float u_ip11 = 0.F;
    float _d_u_im11 = 0.F;
    float u_im11 = 0.F;
    float _d_v_jp11 = 0.F;
    float v_jp11 = 0.F;
    float _d_v_jm11 = 0.F;
    float v_jm11 = 0.F;
    float _d_p_kp1 = 0.F;
    float p_kp1 = 0.F;
    float _d_p_ijk1 = 0.F;
    float p_ijk1 = 0.F;
    float _d_conv_x1 = 0.F;
    float conv_x1 = 0.F;
    float _d_conv_y1 = 0.F;
    float conv_y1 = 0.F;
    float _d_conv_z1 = 0.F;
    float conv_z1 = 0.F;
    float _d_pres1 = 0.F;
    float pres1 = 0.F;
    float _d_diff1 = 0.F;
    float diff1 = 0.F;
    bool _cond3;
    float _d_u_ip12 = 0.F;
    float u_ip12 = 0.F;
    float _d_u_im12 = 0.F;
    float u_im12 = 0.F;
    float _d_v_jp12 = 0.F;
    float v_jp12 = 0.F;
    float _d_v_jm12 = 0.F;
    float v_jm12 = 0.F;
    float _d_w_kp12 = 0.F;
    float w_kp12 = 0.F;
    float _d_w_km12 = 0.F;
    float w_km12 = 0.F;
    float _d_cont = 0.F;
    float cont = 0.F;
    float _d_result = 0.F;
    float result = float(0.);
    {
        _cond0 = eq_type == 0;
        if (_cond0) {
            u_ip1 = get_field_value(y, 0, i + 1, j, k, xN, yN, zN, u_inlet);
            u_im1 = get_field_value(y, 0, i - 1, j, k, xN, yN, zN, u_inlet);
            u_jp1 = get_field_value(y, 0, i, j + 1, k, xN, yN, zN, u_inlet);
            u_jm1 = get_field_value(y, 0, i, j - 1, k, xN, yN, zN, u_inlet);
            u_kp1 = get_field_value(y, 0, i, j, k + 1, xN, yN, zN, u_inlet);
            u_km1 = get_field_value(y, 0, i, j, k - 1, xN, yN, zN, u_inlet);
            u_ijk = get_field_value(y, 0, i, j, k, xN, yN, zN, u_inlet);
            v_jp1 = get_field_value(y, 1, i, j + 1, k, xN, yN, zN, u_inlet);
            v_jm1 = get_field_value(y, 1, i, j - 1, k, xN, yN, zN, u_inlet);
            w_kp1 = get_field_value(y, 2, i, j, k + 1, xN, yN, zN, u_inlet);
            w_km1 = get_field_value(y, 2, i, j, k - 1, xN, yN, zN, u_inlet);
            p_ip1 = get_field_value(y, 3, i + 1, j, k, xN, yN, zN, u_inlet);
            p_ijk = get_field_value(y, 3, i, j, k, xN, yN, zN, u_inlet);
            conv_x = 0.5 * dy * dz * (u_ip1 * u_ip1 - u_im1 * u_im1);
            conv_y = 0.5 * dx * dz * (u_jp1 * v_jp1 - u_jm1 * v_jm1);
            conv_z = 0.5 * dx * dy * (u_kp1 * w_kp1 - u_km1 * w_km1);
            pres = (dy * dz) * (p_ip1 - p_ijk);
            diff = (1. / Re) * ((dy * dz / dx) * (u_ip1 - 2. * u_ijk + u_im1) + (dx * dz / dy) * (u_jp1 - 2. * u_ijk + u_jm1) + (dx * dy / dz) * (u_kp1 - 2. * u_ijk + u_km1));
            result = conv_x + conv_y + conv_z + pres - diff;
        } else {
            _cond1 = eq_type == 1;
            if (_cond1) {
                v_ip1 = get_field_value(y, 1, i + 1, j, k, xN, yN, zN, u_inlet);
                v_im1 = get_field_value(y, 1, i - 1, j, k, xN, yN, zN, u_inlet);
                v_jp10 = get_field_value(y, 1, i, j + 1, k, xN, yN, zN, u_inlet);
                v_jm10 = get_field_value(y, 1, i, j - 1, k, xN, yN, zN, u_inlet);
                v_kp1 = get_field_value(y, 1, i, j, k + 1, xN, yN, zN, u_inlet);
                v_km1 = get_field_value(y, 1, i, j, k - 1, xN, yN, zN, u_inlet);
                v_ijk = get_field_value(y, 1, i, j, k, xN, yN, zN, u_inlet);
                u_ip10 = get_field_value(y, 0, i + 1, j, k, xN, yN, zN, u_inlet);
                u_im10 = get_field_value(y, 0, i - 1, j, k, xN, yN, zN, u_inlet);
                w_kp10 = get_field_value(y, 2, i, j, k + 1, xN, yN, zN, u_inlet);
                w_km10 = get_field_value(y, 2, i, j, k - 1, xN, yN, zN, u_inlet);
                p_jp1 = get_field_value(y, 3, i, j + 1, k, xN, yN, zN, u_inlet);
                p_ijk0 = get_field_value(y, 3, i, j, k, xN, yN, zN, u_inlet);
                conv_x0 = 0.5 * dy * dz * (u_ip10 * v_ip1 - u_im10 * v_im1);
                conv_y0 = 0.5 * dx * dz * (v_jp10 * v_jp10 - v_jm10 * v_jm10);
                conv_z0 = 0.5 * dx * dy * (v_kp1 * w_kp10 - v_km1 * w_km10);
                pres0 = (dx * dz) * (p_jp1 - p_ijk0);
                diff0 = (1. / Re) * ((dy * dz / dx) * (v_ip1 - 2. * v_ijk + v_im1) + (dx * dz / dy) * (v_jp10 - 2. * v_ijk + v_jm10) + (dx * dy / dz) * (v_kp1 - 2. * v_ijk + v_km1));
                result = conv_x0 + conv_y0 + conv_z0 + pres0 - diff0;
            } else {
                _cond2 = eq_type == 2;
                if (_cond2) {
                    w_ip1 = get_field_value(y, 2, i + 1, j, k, xN, yN, zN, u_inlet);
                    w_im1 = get_field_value(y, 2, i - 1, j, k, xN, yN, zN, u_inlet);
                    w_jp1 = get_field_value(y, 2, i, j + 1, k, xN, yN, zN, u_inlet);
                    w_jm1 = get_field_value(y, 2, i, j - 1, k, xN, yN, zN, u_inlet);
                    w_kp11 = get_field_value(y, 2, i, j, k + 1, xN, yN, zN, u_inlet);
                    w_km11 = get_field_value(y, 2, i, j, k - 1, xN, yN, zN, u_inlet);
                    w_ijk = get_field_value(y, 2, i, j, k, xN, yN, zN, u_inlet);
                    u_ip11 = get_field_value(y, 0, i + 1, j, k, xN, yN, zN, u_inlet);
                    u_im11 = get_field_value(y, 0, i - 1, j, k, xN, yN, zN, u_inlet);
                    v_jp11 = get_field_value(y, 1, i, j + 1, k, xN, yN, zN, u_inlet);
                    v_jm11 = get_field_value(y, 1, i, j - 1, k, xN, yN, zN, u_inlet);
                    p_kp1 = get_field_value(y, 3, i, j, k + 1, xN, yN, zN, u_inlet);
                    p_ijk1 = get_field_value(y, 3, i, j, k, xN, yN, zN, u_inlet);
                    conv_x1 = 0.5 * dy * dz * (u_ip11 * w_ip1 - u_im11 * w_im1);
                    conv_y1 = 0.5 * dx * dz * (v_jp11 * w_jp1 - v_jm11 * w_jm1);
                    conv_z1 = 0.5 * dx * dy * (w_kp11 * w_kp11 - w_km11 * w_km11);
                    pres1 = (dx * dy) * (p_kp1 - p_ijk1);
                    diff1 = (1. / Re) * ((dy * dz / dx) * (w_ip1 - 2. * w_ijk + w_im1) + (dx * dz / dy) * (w_jp1 - 2. * w_ijk + w_jm1) + (dx * dy / dz) * (w_kp11 - 2. * w_ijk + w_km11));
                    result = conv_x1 + conv_y1 + conv_z1 + pres1 - diff1;
                } else {
                    _cond3 = eq_type == 3;
                    if (_cond3) {
                        u_ip12 = get_field_value(y, 0, i + 1, j, k, xN, yN, zN, u_inlet);
                        u_im12 = get_field_value(y, 0, i - 1, j, k, xN, yN, zN, u_inlet);
                        v_jp12 = get_field_value(y, 1, i, j + 1, k, xN, yN, zN, u_inlet);
                        v_jm12 = get_field_value(y, 1, i, j - 1, k, xN, yN, zN, u_inlet);
                        w_kp12 = get_field_value(y, 2, i, j, k + 1, xN, yN, zN, u_inlet);
                        w_km12 = get_field_value(y, 2, i, j, k - 1, xN, yN, zN, u_inlet);
                        cont = (dy * dz / 2.) * (u_ip12 - u_im12) + (dx * dz / 2.) * (v_jp12 - v_jm12) + (dx * dy / 2.) * (w_kp12 - w_km12);
                        result = cont;
                    }
                }
            }
        }
    }
    _d_result += 1;
    if (_cond0) {
        {
            _d_conv_x += _d_result;
            _d_conv_y += _d_result;
            _d_conv_z += _d_result;
            _d_pres += _d_result;
            _d_diff += -_d_result;
            _d_result = 0.F;
        }
        {
            double _r91 = _d_diff * ((dy * dz / dx) * (u_ip1 - 2. * u_ijk + u_im1) + (dx * dz / dy) * (u_jp1 - 2. * u_ijk + u_jm1) + (dx * dy / dz) * (u_kp1 - 2. * u_ijk + u_km1)) * -(1. / (Re * Re));
            _d_Re += _r91;
            _d_dy += (1. / Re) * _d_diff * (u_ip1 - 2. * u_ijk + u_im1) / dx * dz;
            _d_dz += dy * (1. / Re) * _d_diff * (u_ip1 - 2. * u_ijk + u_im1) / dx;
            double _r92 = (1. / Re) * _d_diff * (u_ip1 - 2. * u_ijk + u_im1) * -(dy * dz / (dx * dx));
            _d_dx += _r92;
            _d_u_ip1 += (dy * dz / dx) * (1. / Re) * _d_diff;
            _d_u_ijk += 2. * -(dy * dz / dx) * (1. / Re) * _d_diff;
            _d_u_im1 += (dy * dz / dx) * (1. / Re) * _d_diff;
            _d_dx += (1. / Re) * _d_diff * (u_jp1 - 2. * u_ijk + u_jm1) / dy * dz;
            _d_dz += dx * (1. / Re) * _d_diff * (u_jp1 - 2. * u_ijk + u_jm1) / dy;
            double _r93 = (1. / Re) * _d_diff * (u_jp1 - 2. * u_ijk + u_jm1) * -(dx * dz / (dy * dy));
            _d_dy += _r93;
            _d_u_jp1 += (dx * dz / dy) * (1. / Re) * _d_diff;
            _d_u_ijk += 2. * -(dx * dz / dy) * (1. / Re) * _d_diff;
            _d_u_jm1 += (dx * dz / dy) * (1. / Re) * _d_diff;
            _d_dx += (1. / Re) * _d_diff * (u_kp1 - 2. * u_ijk + u_km1) / dz * dy;
            _d_dy += dx * (1. / Re) * _d_diff * (u_kp1 - 2. * u_ijk + u_km1) / dz;
            double _r94 = (1. / Re) * _d_diff * (u_kp1 - 2. * u_ijk + u_km1) * -(dx * dy / (dz * dz));
            _d_dz += _r94;
            _d_u_kp1 += (dx * dy / dz) * (1. / Re) * _d_diff;
            _d_u_ijk += 2. * -(dx * dy / dz) * (1. / Re) * _d_diff;
            _d_u_km1 += (dx * dy / dz) * (1. / Re) * _d_diff;
        }
        {
            _d_dy += _d_pres * (p_ip1 - p_ijk) * dz;
            _d_dz += dy * _d_pres * (p_ip1 - p_ijk);
            _d_p_ip1 += (dy * dz) * _d_pres;
            _d_p_ijk += -(dy * dz) * _d_pres;
        }
        {
            _d_dx += 0.5 * _d_conv_z * (u_kp1 * w_kp1 - u_km1 * w_km1) * dy;
            _d_dy += 0.5 * dx * _d_conv_z * (u_kp1 * w_kp1 - u_km1 * w_km1);
            _d_u_kp1 += 0.5 * dx * dy * _d_conv_z * w_kp1;
            _d_w_kp1 += u_kp1 * 0.5 * dx * dy * _d_conv_z;
            _d_u_km1 += -0.5 * dx * dy * _d_conv_z * w_km1;
            _d_w_km1 += u_km1 * -0.5 * dx * dy * _d_conv_z;
        }
        {
            _d_dx += 0.5 * _d_conv_y * (u_jp1 * v_jp1 - u_jm1 * v_jm1) * dz;
            _d_dz += 0.5 * dx * _d_conv_y * (u_jp1 * v_jp1 - u_jm1 * v_jm1);
            _d_u_jp1 += 0.5 * dx * dz * _d_conv_y * v_jp1;
            _d_v_jp1 += u_jp1 * 0.5 * dx * dz * _d_conv_y;
            _d_u_jm1 += -0.5 * dx * dz * _d_conv_y * v_jm1;
            _d_v_jm1 += u_jm1 * -0.5 * dx * dz * _d_conv_y;
        }
        {
            _d_dy += 0.5 * _d_conv_x * (u_ip1 * u_ip1 - u_im1 * u_im1) * dz;
            _d_dz += 0.5 * dy * _d_conv_x * (u_ip1 * u_ip1 - u_im1 * u_im1);
            _d_u_ip1 += 0.5 * dy * dz * _d_conv_x * u_ip1;
            _d_u_ip1 += u_ip1 * 0.5 * dy * dz * _d_conv_x;
            _d_u_im1 += -0.5 * dy * dz * _d_conv_x * u_im1;
            _d_u_im1 += u_im1 * -0.5 * dy * dz * _d_conv_x;
        }
        {
            int _r84 = 0;
            int _r85 = 0;
            int _r86 = 0;
            int _r87 = 0;
            int _r88 = 0;
            int _r89 = 0;
            int _r90 = 0;
            get_field_value_pullback(y, 3, i, j, k, xN, yN, zN, u_inlet, _d_p_ijk, _d_y, &_r84, &_r85, &_r86, &_r87, &_r88, &_r89, &_r90);
            _d_i += _r85;
            _d_j += _r86;
            _d_k += _r87;
            _d_xN += _r88;
            _d_yN += _r89;
            _d_zN += _r90;
        }
        {
            int _r77 = 0;
            int _r78 = 0;
            int _r79 = 0;
            int _r80 = 0;
            int _r81 = 0;
            int _r82 = 0;
            int _r83 = 0;
            get_field_value_pullback(y, 3, i + 1, j, k, xN, yN, zN, u_inlet, _d_p_ip1, _d_y, &_r77, &_r78, &_r79, &_r80, &_r81, &_r82, &_r83);
            _d_i += _r78;
            _d_j += _r79;
            _d_k += _r80;
            _d_xN += _r81;
            _d_yN += _r82;
            _d_zN += _r83;
        }
        {
            int _r70 = 0;
            int _r71 = 0;
            int _r72 = 0;
            int _r73 = 0;
            int _r74 = 0;
            int _r75 = 0;
            int _r76 = 0;
            get_field_value_pullback(y, 2, i, j, k - 1, xN, yN, zN, u_inlet, _d_w_km1, _d_y, &_r70, &_r71, &_r72, &_r73, &_r74, &_r75, &_r76);
            _d_i += _r71;
            _d_j += _r72;
            _d_k += _r73;
            _d_xN += _r74;
            _d_yN += _r75;
            _d_zN += _r76;
        }
        {
            int _r63 = 0;
            int _r64 = 0;
            int _r65 = 0;
            int _r66 = 0;
            int _r67 = 0;
            int _r68 = 0;
            int _r69 = 0;
            get_field_value_pullback(y, 2, i, j, k + 1, xN, yN, zN, u_inlet, _d_w_kp1, _d_y, &_r63, &_r64, &_r65, &_r66, &_r67, &_r68, &_r69);
            _d_i += _r64;
            _d_j += _r65;
            _d_k += _r66;
            _d_xN += _r67;
            _d_yN += _r68;
            _d_zN += _r69;
        }
        {
            int _r56 = 0;
            int _r57 = 0;
            int _r58 = 0;
            int _r59 = 0;
            int _r60 = 0;
            int _r61 = 0;
            int _r62 = 0;
            get_field_value_pullback(y, 1, i, j - 1, k, xN, yN, zN, u_inlet, _d_v_jm1, _d_y, &_r56, &_r57, &_r58, &_r59, &_r60, &_r61, &_r62);
            _d_i += _r57;
            _d_j += _r58;
            _d_k += _r59;
            _d_xN += _r60;
            _d_yN += _r61;
            _d_zN += _r62;
        }
        {
            int _r49 = 0;
            int _r50 = 0;
            int _r51 = 0;
            int _r52 = 0;
            int _r53 = 0;
            int _r54 = 0;
            int _r55 = 0;
            get_field_value_pullback(y, 1, i, j + 1, k, xN, yN, zN, u_inlet, _d_v_jp1, _d_y, &_r49, &_r50, &_r51, &_r52, &_r53, &_r54, &_r55);
            _d_i += _r50;
            _d_j += _r51;
            _d_k += _r52;
            _d_xN += _r53;
            _d_yN += _r54;
            _d_zN += _r55;
        }
        {
            int _r42 = 0;
            int _r43 = 0;
            int _r44 = 0;
            int _r45 = 0;
            int _r46 = 0;
            int _r47 = 0;
            int _r48 = 0;
            get_field_value_pullback(y, 0, i, j, k, xN, yN, zN, u_inlet, _d_u_ijk, _d_y, &_r42, &_r43, &_r44, &_r45, &_r46, &_r47, &_r48);
            _d_i += _r43;
            _d_j += _r44;
            _d_k += _r45;
            _d_xN += _r46;
            _d_yN += _r47;
            _d_zN += _r48;
        }
        {
            int _r35 = 0;
            int _r36 = 0;
            int _r37 = 0;
            int _r38 = 0;
            int _r39 = 0;
            int _r40 = 0;
            int _r41 = 0;
            get_field_value_pullback(y, 0, i, j, k - 1, xN, yN, zN, u_inlet, _d_u_km1, _d_y, &_r35, &_r36, &_r37, &_r38, &_r39, &_r40, &_r41);
            _d_i += _r36;
            _d_j += _r37;
            _d_k += _r38;
            _d_xN += _r39;
            _d_yN += _r40;
            _d_zN += _r41;
        }
        {
            int _r28 = 0;
            int _r29 = 0;
            int _r30 = 0;
            int _r31 = 0;
            int _r32 = 0;
            int _r33 = 0;
            int _r34 = 0;
            get_field_value_pullback(y, 0, i, j, k + 1, xN, yN, zN, u_inlet, _d_u_kp1, _d_y, &_r28, &_r29, &_r30, &_r31, &_r32, &_r33, &_r34);
            _d_i += _r29;
            _d_j += _r30;
            _d_k += _r31;
            _d_xN += _r32;
            _d_yN += _r33;
            _d_zN += _r34;
        }
        {
            int _r21 = 0;
            int _r22 = 0;
            int _r23 = 0;
            int _r24 = 0;
            int _r25 = 0;
            int _r26 = 0;
            int _r27 = 0;
            get_field_value_pullback(y, 0, i, j - 1, k, xN, yN, zN, u_inlet, _d_u_jm1, _d_y, &_r21, &_r22, &_r23, &_r24, &_r25, &_r26, &_r27);
            _d_i += _r22;
            _d_j += _r23;
            _d_k += _r24;
            _d_xN += _r25;
            _d_yN += _r26;
            _d_zN += _r27;
        }
        {
            int _r14 = 0;
            int _r15 = 0;
            int _r16 = 0;
            int _r17 = 0;
            int _r18 = 0;
            int _r19 = 0;
            int _r20 = 0;
            get_field_value_pullback(y, 0, i, j + 1, k, xN, yN, zN, u_inlet, _d_u_jp1, _d_y, &_r14, &_r15, &_r16, &_r17, &_r18, &_r19, &_r20);
            _d_i += _r15;
            _d_j += _r16;
            _d_k += _r17;
            _d_xN += _r18;
            _d_yN += _r19;
            _d_zN += _r20;
        }
        {
            int _r7 = 0;
            int _r8 = 0;
            int _r9 = 0;
            int _r10 = 0;
            int _r11 = 0;
            int _r12 = 0;
            int _r13 = 0;
            get_field_value_pullback(y, 0, i - 1, j, k, xN, yN, zN, u_inlet, _d_u_im1, _d_y, &_r7, &_r8, &_r9, &_r10, &_r11, &_r12, &_r13);
            _d_i += _r8;
            _d_j += _r9;
            _d_k += _r10;
            _d_xN += _r11;
            _d_yN += _r12;
            _d_zN += _r13;
        }
        {
            int _r0 = 0;
            int _r1 = 0;
            int _r2 = 0;
            int _r3 = 0;
            int _r4 = 0;
            int _r5 = 0;
            int _r6 = 0;
            get_field_value_pullback(y, 0, i + 1, j, k, xN, yN, zN, u_inlet, _d_u_ip1, _d_y, &_r0, &_r1, &_r2, &_r3, &_r4, &_r5, &_r6);
            _d_i += _r1;
            _d_j += _r2;
            _d_k += _r3;
            _d_xN += _r4;
            _d_yN += _r5;
            _d_zN += _r6;
        }
    } else if (_cond1) {
        {
            _d_conv_x0 += _d_result;
            _d_conv_y0 += _d_result;
            _d_conv_z0 += _d_result;
            _d_pres0 += _d_result;
            _d_diff0 += -_d_result;
            _d_result = 0.F;
        }
        {
            double _r186 = _d_diff0 * ((dy * dz / dx) * (v_ip1 - 2. * v_ijk + v_im1) + (dx * dz / dy) * (v_jp10 - 2. * v_ijk + v_jm10) + (dx * dy / dz) * (v_kp1 - 2. * v_ijk + v_km1)) * -(1. / (Re * Re));
            _d_Re += _r186;
            _d_dy += (1. / Re) * _d_diff0 * (v_ip1 - 2. * v_ijk + v_im1) / dx * dz;
            _d_dz += dy * (1. / Re) * _d_diff0 * (v_ip1 - 2. * v_ijk + v_im1) / dx;
            double _r187 = (1. / Re) * _d_diff0 * (v_ip1 - 2. * v_ijk + v_im1) * -(dy * dz / (dx * dx));
            _d_dx += _r187;
            _d_v_ip1 += (dy * dz / dx) * (1. / Re) * _d_diff0;
            _d_v_ijk += 2. * -(dy * dz / dx) * (1. / Re) * _d_diff0;
            _d_v_im1 += (dy * dz / dx) * (1. / Re) * _d_diff0;
            _d_dx += (1. / Re) * _d_diff0 * (v_jp10 - 2. * v_ijk + v_jm10) / dy * dz;
            _d_dz += dx * (1. / Re) * _d_diff0 * (v_jp10 - 2. * v_ijk + v_jm10) / dy;
            double _r188 = (1. / Re) * _d_diff0 * (v_jp10 - 2. * v_ijk + v_jm10) * -(dx * dz / (dy * dy));
            _d_dy += _r188;
            _d_v_jp10 += (dx * dz / dy) * (1. / Re) * _d_diff0;
            _d_v_ijk += 2. * -(dx * dz / dy) * (1. / Re) * _d_diff0;
            _d_v_jm10 += (dx * dz / dy) * (1. / Re) * _d_diff0;
            _d_dx += (1. / Re) * _d_diff0 * (v_kp1 - 2. * v_ijk + v_km1) / dz * dy;
            _d_dy += dx * (1. / Re) * _d_diff0 * (v_kp1 - 2. * v_ijk + v_km1) / dz;
            double _r189 = (1. / Re) * _d_diff0 * (v_kp1 - 2. * v_ijk + v_km1) * -(dx * dy / (dz * dz));
            _d_dz += _r189;
            _d_v_kp1 += (dx * dy / dz) * (1. / Re) * _d_diff0;
            _d_v_ijk += 2. * -(dx * dy / dz) * (1. / Re) * _d_diff0;
            _d_v_km1 += (dx * dy / dz) * (1. / Re) * _d_diff0;
        }
        {
            _d_dx += _d_pres0 * (p_jp1 - p_ijk0) * dz;
            _d_dz += dx * _d_pres0 * (p_jp1 - p_ijk0);
            _d_p_jp1 += (dx * dz) * _d_pres0;
            _d_p_ijk0 += -(dx * dz) * _d_pres0;
        }
        {
            _d_dx += 0.5 * _d_conv_z0 * (v_kp1 * w_kp10 - v_km1 * w_km10) * dy;
            _d_dy += 0.5 * dx * _d_conv_z0 * (v_kp1 * w_kp10 - v_km1 * w_km10);
            _d_v_kp1 += 0.5 * dx * dy * _d_conv_z0 * w_kp10;
            _d_w_kp10 += v_kp1 * 0.5 * dx * dy * _d_conv_z0;
            _d_v_km1 += -0.5 * dx * dy * _d_conv_z0 * w_km10;
            _d_w_km10 += v_km1 * -0.5 * dx * dy * _d_conv_z0;
        }
        {
            _d_dx += 0.5 * _d_conv_y0 * (v_jp10 * v_jp10 - v_jm10 * v_jm10) * dz;
            _d_dz += 0.5 * dx * _d_conv_y0 * (v_jp10 * v_jp10 - v_jm10 * v_jm10);
            _d_v_jp10 += 0.5 * dx * dz * _d_conv_y0 * v_jp10;
            _d_v_jp10 += v_jp10 * 0.5 * dx * dz * _d_conv_y0;
            _d_v_jm10 += -0.5 * dx * dz * _d_conv_y0 * v_jm10;
            _d_v_jm10 += v_jm10 * -0.5 * dx * dz * _d_conv_y0;
        }
        {
            _d_dy += 0.5 * _d_conv_x0 * (u_ip10 * v_ip1 - u_im10 * v_im1) * dz;
            _d_dz += 0.5 * dy * _d_conv_x0 * (u_ip10 * v_ip1 - u_im10 * v_im1);
            _d_u_ip10 += 0.5 * dy * dz * _d_conv_x0 * v_ip1;
            _d_v_ip1 += u_ip10 * 0.5 * dy * dz * _d_conv_x0;
            _d_u_im10 += -0.5 * dy * dz * _d_conv_x0 * v_im1;
            _d_v_im1 += u_im10 * -0.5 * dy * dz * _d_conv_x0;
        }
        {
            int _r179 = 0;
            int _r180 = 0;
            int _r181 = 0;
            int _r182 = 0;
            int _r183 = 0;
            int _r184 = 0;
            int _r185 = 0;
            get_field_value_pullback(y, 3, i, j, k, xN, yN, zN, u_inlet, _d_p_ijk0, _d_y, &_r179, &_r180, &_r181, &_r182, &_r183, &_r184, &_r185);
            _d_i += _r180;
            _d_j += _r181;
            _d_k += _r182;
            _d_xN += _r183;
            _d_yN += _r184;
            _d_zN += _r185;
        }
        {
            int _r172 = 0;
            int _r173 = 0;
            int _r174 = 0;
            int _r175 = 0;
            int _r176 = 0;
            int _r177 = 0;
            int _r178 = 0;
            get_field_value_pullback(y, 3, i, j + 1, k, xN, yN, zN, u_inlet, _d_p_jp1, _d_y, &_r172, &_r173, &_r174, &_r175, &_r176, &_r177, &_r178);
            _d_i += _r173;
            _d_j += _r174;
            _d_k += _r175;
            _d_xN += _r176;
            _d_yN += _r177;
            _d_zN += _r178;
        }
        {
            int _r165 = 0;
            int _r166 = 0;
            int _r167 = 0;
            int _r168 = 0;
            int _r169 = 0;
            int _r170 = 0;
            int _r171 = 0;
            get_field_value_pullback(y, 2, i, j, k - 1, xN, yN, zN, u_inlet, _d_w_km10, _d_y, &_r165, &_r166, &_r167, &_r168, &_r169, &_r170, &_r171);
            _d_i += _r166;
            _d_j += _r167;
            _d_k += _r168;
            _d_xN += _r169;
            _d_yN += _r170;
            _d_zN += _r171;
        }
        {
            int _r158 = 0;
            int _r159 = 0;
            int _r160 = 0;
            int _r161 = 0;
            int _r162 = 0;
            int _r163 = 0;
            int _r164 = 0;
            get_field_value_pullback(y, 2, i, j, k + 1, xN, yN, zN, u_inlet, _d_w_kp10, _d_y, &_r158, &_r159, &_r160, &_r161, &_r162, &_r163, &_r164);
            _d_i += _r159;
            _d_j += _r160;
            _d_k += _r161;
            _d_xN += _r162;
            _d_yN += _r163;
            _d_zN += _r164;
        }
        {
            int _r151 = 0;
            int _r152 = 0;
            int _r153 = 0;
            int _r154 = 0;
            int _r155 = 0;
            int _r156 = 0;
            int _r157 = 0;
            get_field_value_pullback(y, 0, i - 1, j, k, xN, yN, zN, u_inlet, _d_u_im10, _d_y, &_r151, &_r152, &_r153, &_r154, &_r155, &_r156, &_r157);
            _d_i += _r152;
            _d_j += _r153;
            _d_k += _r154;
            _d_xN += _r155;
            _d_yN += _r156;
            _d_zN += _r157;
        }
        {
            int _r144 = 0;
            int _r145 = 0;
            int _r146 = 0;
            int _r147 = 0;
            int _r148 = 0;
            int _r149 = 0;
            int _r150 = 0;
            get_field_value_pullback(y, 0, i + 1, j, k, xN, yN, zN, u_inlet, _d_u_ip10, _d_y, &_r144, &_r145, &_r146, &_r147, &_r148, &_r149, &_r150);
            _d_i += _r145;
            _d_j += _r146;
            _d_k += _r147;
            _d_xN += _r148;
            _d_yN += _r149;
            _d_zN += _r150;
        }
        {
            int _r137 = 0;
            int _r138 = 0;
            int _r139 = 0;
            int _r140 = 0;
            int _r141 = 0;
            int _r142 = 0;
            int _r143 = 0;
            get_field_value_pullback(y, 1, i, j, k, xN, yN, zN, u_inlet, _d_v_ijk, _d_y, &_r137, &_r138, &_r139, &_r140, &_r141, &_r142, &_r143);
            _d_i += _r138;
            _d_j += _r139;
            _d_k += _r140;
            _d_xN += _r141;
            _d_yN += _r142;
            _d_zN += _r143;
        }
        {
            int _r130 = 0;
            int _r131 = 0;
            int _r132 = 0;
            int _r133 = 0;
            int _r134 = 0;
            int _r135 = 0;
            int _r136 = 0;
            get_field_value_pullback(y, 1, i, j, k - 1, xN, yN, zN, u_inlet, _d_v_km1, _d_y, &_r130, &_r131, &_r132, &_r133, &_r134, &_r135, &_r136);
            _d_i += _r131;
            _d_j += _r132;
            _d_k += _r133;
            _d_xN += _r134;
            _d_yN += _r135;
            _d_zN += _r136;
        }
        {
            int _r123 = 0;
            int _r124 = 0;
            int _r125 = 0;
            int _r126 = 0;
            int _r127 = 0;
            int _r128 = 0;
            int _r129 = 0;
            get_field_value_pullback(y, 1, i, j, k + 1, xN, yN, zN, u_inlet, _d_v_kp1, _d_y, &_r123, &_r124, &_r125, &_r126, &_r127, &_r128, &_r129);
            _d_i += _r124;
            _d_j += _r125;
            _d_k += _r126;
            _d_xN += _r127;
            _d_yN += _r128;
            _d_zN += _r129;
        }
        {
            int _r116 = 0;
            int _r117 = 0;
            int _r118 = 0;
            int _r119 = 0;
            int _r120 = 0;
            int _r121 = 0;
            int _r122 = 0;
            get_field_value_pullback(y, 1, i, j - 1, k, xN, yN, zN, u_inlet, _d_v_jm10, _d_y, &_r116, &_r117, &_r118, &_r119, &_r120, &_r121, &_r122);
            _d_i += _r117;
            _d_j += _r118;
            _d_k += _r119;
            _d_xN += _r120;
            _d_yN += _r121;
            _d_zN += _r122;
        }
        {
            int _r109 = 0;
            int _r110 = 0;
            int _r111 = 0;
            int _r112 = 0;
            int _r113 = 0;
            int _r114 = 0;
            int _r115 = 0;
            get_field_value_pullback(y, 1, i, j + 1, k, xN, yN, zN, u_inlet, _d_v_jp10, _d_y, &_r109, &_r110, &_r111, &_r112, &_r113, &_r114, &_r115);
            _d_i += _r110;
            _d_j += _r111;
            _d_k += _r112;
            _d_xN += _r113;
            _d_yN += _r114;
            _d_zN += _r115;
        }
        {
            int _r102 = 0;
            int _r103 = 0;
            int _r104 = 0;
            int _r105 = 0;
            int _r106 = 0;
            int _r107 = 0;
            int _r108 = 0;
            get_field_value_pullback(y, 1, i - 1, j, k, xN, yN, zN, u_inlet, _d_v_im1, _d_y, &_r102, &_r103, &_r104, &_r105, &_r106, &_r107, &_r108);
            _d_i += _r103;
            _d_j += _r104;
            _d_k += _r105;
            _d_xN += _r106;
            _d_yN += _r107;
            _d_zN += _r108;
        }
        {
            int _r95 = 0;
            int _r96 = 0;
            int _r97 = 0;
            int _r98 = 0;
            int _r99 = 0;
            int _r100 = 0;
            int _r101 = 0;
            get_field_value_pullback(y, 1, i + 1, j, k, xN, yN, zN, u_inlet, _d_v_ip1, _d_y, &_r95, &_r96, &_r97, &_r98, &_r99, &_r100, &_r101);
            _d_i += _r96;
            _d_j += _r97;
            _d_k += _r98;
            _d_xN += _r99;
            _d_yN += _r100;
            _d_zN += _r101;
        }
    } else if (_cond2) {
        {
            _d_conv_x1 += _d_result;
            _d_conv_y1 += _d_result;
            _d_conv_z1 += _d_result;
            _d_pres1 += _d_result;
            _d_diff1 += -_d_result;
            _d_result = 0.F;
        }
        {
            double _r281 = _d_diff1 * ((dy * dz / dx) * (w_ip1 - 2. * w_ijk + w_im1) + (dx * dz / dy) * (w_jp1 - 2. * w_ijk + w_jm1) + (dx * dy / dz) * (w_kp11 - 2. * w_ijk + w_km11)) * -(1. / (Re * Re));
            _d_Re += _r281;
            _d_dy += (1. / Re) * _d_diff1 * (w_ip1 - 2. * w_ijk + w_im1) / dx * dz;
            _d_dz += dy * (1. / Re) * _d_diff1 * (w_ip1 - 2. * w_ijk + w_im1) / dx;
            double _r282 = (1. / Re) * _d_diff1 * (w_ip1 - 2. * w_ijk + w_im1) * -(dy * dz / (dx * dx));
            _d_dx += _r282;
            _d_w_ip1 += (dy * dz / dx) * (1. / Re) * _d_diff1;
            _d_w_ijk += 2. * -(dy * dz / dx) * (1. / Re) * _d_diff1;
            _d_w_im1 += (dy * dz / dx) * (1. / Re) * _d_diff1;
            _d_dx += (1. / Re) * _d_diff1 * (w_jp1 - 2. * w_ijk + w_jm1) / dy * dz;
            _d_dz += dx * (1. / Re) * _d_diff1 * (w_jp1 - 2. * w_ijk + w_jm1) / dy;
            double _r283 = (1. / Re) * _d_diff1 * (w_jp1 - 2. * w_ijk + w_jm1) * -(dx * dz / (dy * dy));
            _d_dy += _r283;
            _d_w_jp1 += (dx * dz / dy) * (1. / Re) * _d_diff1;
            _d_w_ijk += 2. * -(dx * dz / dy) * (1. / Re) * _d_diff1;
            _d_w_jm1 += (dx * dz / dy) * (1. / Re) * _d_diff1;
            _d_dx += (1. / Re) * _d_diff1 * (w_kp11 - 2. * w_ijk + w_km11) / dz * dy;
            _d_dy += dx * (1. / Re) * _d_diff1 * (w_kp11 - 2. * w_ijk + w_km11) / dz;
            double _r284 = (1. / Re) * _d_diff1 * (w_kp11 - 2. * w_ijk + w_km11) * -(dx * dy / (dz * dz));
            _d_dz += _r284;
            _d_w_kp11 += (dx * dy / dz) * (1. / Re) * _d_diff1;
            _d_w_ijk += 2. * -(dx * dy / dz) * (1. / Re) * _d_diff1;
            _d_w_km11 += (dx * dy / dz) * (1. / Re) * _d_diff1;
        }
        {
            _d_dx += _d_pres1 * (p_kp1 - p_ijk1) * dy;
            _d_dy += dx * _d_pres1 * (p_kp1 - p_ijk1);
            _d_p_kp1 += (dx * dy) * _d_pres1;
            _d_p_ijk1 += -(dx * dy) * _d_pres1;
        }
        {
            _d_dx += 0.5 * _d_conv_z1 * (w_kp11 * w_kp11 - w_km11 * w_km11) * dy;
            _d_dy += 0.5 * dx * _d_conv_z1 * (w_kp11 * w_kp11 - w_km11 * w_km11);
            _d_w_kp11 += 0.5 * dx * dy * _d_conv_z1 * w_kp11;
            _d_w_kp11 += w_kp11 * 0.5 * dx * dy * _d_conv_z1;
            _d_w_km11 += -0.5 * dx * dy * _d_conv_z1 * w_km11;
            _d_w_km11 += w_km11 * -0.5 * dx * dy * _d_conv_z1;
        }
        {
            _d_dx += 0.5 * _d_conv_y1 * (v_jp11 * w_jp1 - v_jm11 * w_jm1) * dz;
            _d_dz += 0.5 * dx * _d_conv_y1 * (v_jp11 * w_jp1 - v_jm11 * w_jm1);
            _d_v_jp11 += 0.5 * dx * dz * _d_conv_y1 * w_jp1;
            _d_w_jp1 += v_jp11 * 0.5 * dx * dz * _d_conv_y1;
            _d_v_jm11 += -0.5 * dx * dz * _d_conv_y1 * w_jm1;
            _d_w_jm1 += v_jm11 * -0.5 * dx * dz * _d_conv_y1;
        }
        {
            _d_dy += 0.5 * _d_conv_x1 * (u_ip11 * w_ip1 - u_im11 * w_im1) * dz;
            _d_dz += 0.5 * dy * _d_conv_x1 * (u_ip11 * w_ip1 - u_im11 * w_im1);
            _d_u_ip11 += 0.5 * dy * dz * _d_conv_x1 * w_ip1;
            _d_w_ip1 += u_ip11 * 0.5 * dy * dz * _d_conv_x1;
            _d_u_im11 += -0.5 * dy * dz * _d_conv_x1 * w_im1;
            _d_w_im1 += u_im11 * -0.5 * dy * dz * _d_conv_x1;
        }
        {
            int _r274 = 0;
            int _r275 = 0;
            int _r276 = 0;
            int _r277 = 0;
            int _r278 = 0;
            int _r279 = 0;
            int _r280 = 0;
            get_field_value_pullback(y, 3, i, j, k, xN, yN, zN, u_inlet, _d_p_ijk1, _d_y, &_r274, &_r275, &_r276, &_r277, &_r278, &_r279, &_r280);
            _d_i += _r275;
            _d_j += _r276;
            _d_k += _r277;
            _d_xN += _r278;
            _d_yN += _r279;
            _d_zN += _r280;
        }
        {
            int _r267 = 0;
            int _r268 = 0;
            int _r269 = 0;
            int _r270 = 0;
            int _r271 = 0;
            int _r272 = 0;
            int _r273 = 0;
            get_field_value_pullback(y, 3, i, j, k + 1, xN, yN, zN, u_inlet, _d_p_kp1, _d_y, &_r267, &_r268, &_r269, &_r270, &_r271, &_r272, &_r273);
            _d_i += _r268;
            _d_j += _r269;
            _d_k += _r270;
            _d_xN += _r271;
            _d_yN += _r272;
            _d_zN += _r273;
        }
        {
            int _r260 = 0;
            int _r261 = 0;
            int _r262 = 0;
            int _r263 = 0;
            int _r264 = 0;
            int _r265 = 0;
            int _r266 = 0;
            get_field_value_pullback(y, 1, i, j - 1, k, xN, yN, zN, u_inlet, _d_v_jm11, _d_y, &_r260, &_r261, &_r262, &_r263, &_r264, &_r265, &_r266);
            _d_i += _r261;
            _d_j += _r262;
            _d_k += _r263;
            _d_xN += _r264;
            _d_yN += _r265;
            _d_zN += _r266;
        }
        {
            int _r253 = 0;
            int _r254 = 0;
            int _r255 = 0;
            int _r256 = 0;
            int _r257 = 0;
            int _r258 = 0;
            int _r259 = 0;
            get_field_value_pullback(y, 1, i, j + 1, k, xN, yN, zN, u_inlet, _d_v_jp11, _d_y, &_r253, &_r254, &_r255, &_r256, &_r257, &_r258, &_r259);
            _d_i += _r254;
            _d_j += _r255;
            _d_k += _r256;
            _d_xN += _r257;
            _d_yN += _r258;
            _d_zN += _r259;
        }
        {
            int _r246 = 0;
            int _r247 = 0;
            int _r248 = 0;
            int _r249 = 0;
            int _r250 = 0;
            int _r251 = 0;
            int _r252 = 0;
            get_field_value_pullback(y, 0, i - 1, j, k, xN, yN, zN, u_inlet, _d_u_im11, _d_y, &_r246, &_r247, &_r248, &_r249, &_r250, &_r251, &_r252);
            _d_i += _r247;
            _d_j += _r248;
            _d_k += _r249;
            _d_xN += _r250;
            _d_yN += _r251;
            _d_zN += _r252;
        }
        {
            int _r239 = 0;
            int _r240 = 0;
            int _r241 = 0;
            int _r242 = 0;
            int _r243 = 0;
            int _r244 = 0;
            int _r245 = 0;
            get_field_value_pullback(y, 0, i + 1, j, k, xN, yN, zN, u_inlet, _d_u_ip11, _d_y, &_r239, &_r240, &_r241, &_r242, &_r243, &_r244, &_r245);
            _d_i += _r240;
            _d_j += _r241;
            _d_k += _r242;
            _d_xN += _r243;
            _d_yN += _r244;
            _d_zN += _r245;
        }
        {
            int _r232 = 0;
            int _r233 = 0;
            int _r234 = 0;
            int _r235 = 0;
            int _r236 = 0;
            int _r237 = 0;
            int _r238 = 0;
            get_field_value_pullback(y, 2, i, j, k, xN, yN, zN, u_inlet, _d_w_ijk, _d_y, &_r232, &_r233, &_r234, &_r235, &_r236, &_r237, &_r238);
            _d_i += _r233;
            _d_j += _r234;
            _d_k += _r235;
            _d_xN += _r236;
            _d_yN += _r237;
            _d_zN += _r238;
        }
        {
            int _r225 = 0;
            int _r226 = 0;
            int _r227 = 0;
            int _r228 = 0;
            int _r229 = 0;
            int _r230 = 0;
            int _r231 = 0;
            get_field_value_pullback(y, 2, i, j, k - 1, xN, yN, zN, u_inlet, _d_w_km11, _d_y, &_r225, &_r226, &_r227, &_r228, &_r229, &_r230, &_r231);
            _d_i += _r226;
            _d_j += _r227;
            _d_k += _r228;
            _d_xN += _r229;
            _d_yN += _r230;
            _d_zN += _r231;
        }
        {
            int _r218 = 0;
            int _r219 = 0;
            int _r220 = 0;
            int _r221 = 0;
            int _r222 = 0;
            int _r223 = 0;
            int _r224 = 0;
            get_field_value_pullback(y, 2, i, j, k + 1, xN, yN, zN, u_inlet, _d_w_kp11, _d_y, &_r218, &_r219, &_r220, &_r221, &_r222, &_r223, &_r224);
            _d_i += _r219;
            _d_j += _r220;
            _d_k += _r221;
            _d_xN += _r222;
            _d_yN += _r223;
            _d_zN += _r224;
        }
        {
            int _r211 = 0;
            int _r212 = 0;
            int _r213 = 0;
            int _r214 = 0;
            int _r215 = 0;
            int _r216 = 0;
            int _r217 = 0;
            get_field_value_pullback(y, 2, i, j - 1, k, xN, yN, zN, u_inlet, _d_w_jm1, _d_y, &_r211, &_r212, &_r213, &_r214, &_r215, &_r216, &_r217);
            _d_i += _r212;
            _d_j += _r213;
            _d_k += _r214;
            _d_xN += _r215;
            _d_yN += _r216;
            _d_zN += _r217;
        }
        {
            int _r204 = 0;
            int _r205 = 0;
            int _r206 = 0;
            int _r207 = 0;
            int _r208 = 0;
            int _r209 = 0;
            int _r210 = 0;
            get_field_value_pullback(y, 2, i, j + 1, k, xN, yN, zN, u_inlet, _d_w_jp1, _d_y, &_r204, &_r205, &_r206, &_r207, &_r208, &_r209, &_r210);
            _d_i += _r205;
            _d_j += _r206;
            _d_k += _r207;
            _d_xN += _r208;
            _d_yN += _r209;
            _d_zN += _r210;
        }
        {
            int _r197 = 0;
            int _r198 = 0;
            int _r199 = 0;
            int _r200 = 0;
            int _r201 = 0;
            int _r202 = 0;
            int _r203 = 0;
            get_field_value_pullback(y, 2, i - 1, j, k, xN, yN, zN, u_inlet, _d_w_im1, _d_y, &_r197, &_r198, &_r199, &_r200, &_r201, &_r202, &_r203);
            _d_i += _r198;
            _d_j += _r199;
            _d_k += _r200;
            _d_xN += _r201;
            _d_yN += _r202;
            _d_zN += _r203;
        }
        {
            int _r190 = 0;
            int _r191 = 0;
            int _r192 = 0;
            int _r193 = 0;
            int _r194 = 0;
            int _r195 = 0;
            int _r196 = 0;
            get_field_value_pullback(y, 2, i + 1, j, k, xN, yN, zN, u_inlet, _d_w_ip1, _d_y, &_r190, &_r191, &_r192, &_r193, &_r194, &_r195, &_r196);
            _d_i += _r191;
            _d_j += _r192;
            _d_k += _r193;
            _d_xN += _r194;
            _d_yN += _r195;
            _d_zN += _r196;
        }
    } else if (_cond3) {
        {
            _d_cont += _d_result;
            _d_result = 0.F;
        }
        {
            _d_dy += _d_cont * (u_ip12 - u_im12) / 2. * dz;
            _d_dz += dy * _d_cont * (u_ip12 - u_im12) / 2.;
            _d_u_ip12 += (dy * dz / 2.) * _d_cont;
            _d_u_im12 += -(dy * dz / 2.) * _d_cont;
            _d_dx += _d_cont * (v_jp12 - v_jm12) / 2. * dz;
            _d_dz += dx * _d_cont * (v_jp12 - v_jm12) / 2.;
            _d_v_jp12 += (dx * dz / 2.) * _d_cont;
            _d_v_jm12 += -(dx * dz / 2.) * _d_cont;
            _d_dx += _d_cont * (w_kp12 - w_km12) / 2. * dy;
            _d_dy += dx * _d_cont * (w_kp12 - w_km12) / 2.;
            _d_w_kp12 += (dx * dy / 2.) * _d_cont;
            _d_w_km12 += -(dx * dy / 2.) * _d_cont;
        }
        {
            int _r320 = 0;
            int _r321 = 0;
            int _r322 = 0;
            int _r323 = 0;
            int _r324 = 0;
            int _r325 = 0;
            int _r326 = 0;
            get_field_value_pullback(y, 2, i, j, k - 1, xN, yN, zN, u_inlet, _d_w_km12, _d_y, &_r320, &_r321, &_r322, &_r323, &_r324, &_r325, &_r326);
            _d_i += _r321;
            _d_j += _r322;
            _d_k += _r323;
            _d_xN += _r324;
            _d_yN += _r325;
            _d_zN += _r326;
        }
        {
            int _r313 = 0;
            int _r314 = 0;
            int _r315 = 0;
            int _r316 = 0;
            int _r317 = 0;
            int _r318 = 0;
            int _r319 = 0;
            get_field_value_pullback(y, 2, i, j, k + 1, xN, yN, zN, u_inlet, _d_w_kp12, _d_y, &_r313, &_r314, &_r315, &_r316, &_r317, &_r318, &_r319);
            _d_i += _r314;
            _d_j += _r315;
            _d_k += _r316;
            _d_xN += _r317;
            _d_yN += _r318;
            _d_zN += _r319;
        }
        {
            int _r306 = 0;
            int _r307 = 0;
            int _r308 = 0;
            int _r309 = 0;
            int _r310 = 0;
            int _r311 = 0;
            int _r312 = 0;
            get_field_value_pullback(y, 1, i, j - 1, k, xN, yN, zN, u_inlet, _d_v_jm12, _d_y, &_r306, &_r307, &_r308, &_r309, &_r310, &_r311, &_r312);
            _d_i += _r307;
            _d_j += _r308;
            _d_k += _r309;
            _d_xN += _r310;
            _d_yN += _r311;
            _d_zN += _r312;
        }
        {
            int _r299 = 0;
            int _r300 = 0;
            int _r301 = 0;
            int _r302 = 0;
            int _r303 = 0;
            int _r304 = 0;
            int _r305 = 0;
            get_field_value_pullback(y, 1, i, j + 1, k, xN, yN, zN, u_inlet, _d_v_jp12, _d_y, &_r299, &_r300, &_r301, &_r302, &_r303, &_r304, &_r305);
            _d_i += _r300;
            _d_j += _r301;
            _d_k += _r302;
            _d_xN += _r303;
            _d_yN += _r304;
            _d_zN += _r305;
        }
        {
            int _r292 = 0;
            int _r293 = 0;
            int _r294 = 0;
            int _r295 = 0;
            int _r296 = 0;
            int _r297 = 0;
            int _r298 = 0;
            get_field_value_pullback(y, 0, i - 1, j, k, xN, yN, zN, u_inlet, _d_u_im12, _d_y, &_r292, &_r293, &_r294, &_r295, &_r296, &_r297, &_r298);
            _d_i += _r293;
            _d_j += _r294;
            _d_k += _r295;
            _d_xN += _r296;
            _d_yN += _r297;
            _d_zN += _r298;
        }
        {
            int _r285 = 0;
            int _r286 = 0;
            int _r287 = 0;
            int _r288 = 0;
            int _r289 = 0;
            int _r290 = 0;
            int _r291 = 0;
            get_field_value_pullback(y, 0, i + 1, j, k, xN, yN, zN, u_inlet, _d_u_ip12, _d_y, &_r285, &_r286, &_r287, &_r288, &_r289, &_r290, &_r291);
            _d_i += _r286;
            _d_j += _r287;
            _d_k += _r288;
            _d_xN += _r289;
            _d_yN += _r290;
            _d_zN += _r291;
        }
    }
}
inline void get_field_value_pullback(const float *y, int field_type, int i, int j, int k, int xN, int yN, int zN, const float *u_inlet, float _d_y0, float *_d_y, int *_d_field_type, int *_d_i, int *_d_j, int *_d_k, int *_d_xN, int *_d_yN, int *_d_zN) {
    bool _cond0;
    bool _cond1;
    bool _cond2;
    bool _cond3;
    bool _cond4;
    bool _cond5;
    bool _cond6;
    bool _cond7;
    bool _cond8;
    bool _cond9;
    bool _cond10;
    bool _cond11;
    int _d_sizeY = 0;
    const int sizeY = yN + 2;
    int _d_sizeZ = 0;
    const int sizeZ = zN + 2;
    int _d_nCell = 0;
    const int nCell = xN * yN * zN;
    int _d_i_clamped = 0;
    int i_clamped = i;
    int _d_j_clamped = 0;
    int j_clamped = j;
    int _d_k_clamped = 0;
    int k_clamped = k;
    {
        _cond0 = i < 1;
        if (_cond0) {
            i_clamped = 1;
            {
                _cond1 = field_type == 0;
                if (_cond1) {
                    goto _label0;
                }
            }
            goto _label1;
        }
    }
    {
        _cond2 = i > xN;
        if (_cond2) {
            i_clamped = xN;
            {
                _cond3 = field_type == 3;
                if (_cond3) {
                    goto _label2;
                }
            }
        }
    }
    {
        _cond4 = j < 1;
        if (_cond4) {
            j_clamped = 1;
            {
                _cond5 = field_type != 3;
                if (_cond5)
                    goto _label3;
            }
        }
    }
    {
        _cond6 = j > yN;
        if (_cond6) {
            j_clamped = yN;
            {
                _cond7 = field_type != 3;
                if (_cond7)
                    goto _label4;
            }
        }
    }
    {
        _cond8 = k < 1;
        if (_cond8) {
            k_clamped = 1;
            {
                _cond9 = field_type != 3;
                if (_cond9)
                    goto _label5;
            }
        }
    }
    {
        _cond10 = k > zN;
        if (_cond10) {
            k_clamped = zN;
            {
                _cond11 = field_type != 3;
                if (_cond11)
                    goto _label6;
            }
        }
    }
    int _d_pos = 0;
    const int pos = (i_clamped - 1) * yN * zN + (j_clamped - 1) * zN + (k_clamped - 1);
    _d_y[field_type * nCell + pos] += _d_y0;
    {
        _d_i_clamped += _d_pos * zN * yN;
        *_d_yN += (i_clamped - 1) * _d_pos * zN;
        *_d_zN += (i_clamped - 1) * yN * _d_pos;
        _d_j_clamped += _d_pos * zN;
        *_d_zN += (j_clamped - 1) * _d_pos;
        _d_k_clamped += _d_pos;
    }
    if (_cond10) {
        if (_cond11)
          _label6:
            ;
        {
            *_d_zN += _d_k_clamped;
            _d_k_clamped = 0;
        }
    }
    if (_cond8) {
        if (_cond9)
          _label5:
            ;
        _d_k_clamped = 0;
    }
    if (_cond6) {
        if (_cond7)
          _label4:
            ;
        {
            *_d_yN += _d_j_clamped;
            _d_j_clamped = 0;
        }
    }
    if (_cond4) {
        if (_cond5)
          _label3:
            ;
        _d_j_clamped = 0;
    }
    if (_cond2) {
        if (_cond3) {
          _label2:
            ;
        }
        {
            *_d_xN += _d_i_clamped;
            _d_i_clamped = 0;
        }
    }
    if (_cond0) {
      _label1:
        ;
        if (_cond1) {
          _label0:
            ;
        }
        _d_i_clamped = 0;
    }
    *_d_k += _d_k_clamped;
    *_d_j += _d_j_clamped;
    *_d_i += _d_i_clamped;
    {
        *_d_xN += _d_nCell * zN * yN;
        *_d_yN += xN * _d_nCell * zN;
        *_d_zN += xN * yN * _d_nCell;
    }
    *_d_zN += _d_sizeZ;
    *_d_yN += _d_sizeY;
}
void compute_residual_component_grad_1(float Re, const float *y, int i, int j, int k, int eq_type, int xN, int yN, int zN, float dx, float dy, float dz, const float *u_inlet, float *_d_y) {
    float _d_Re = 0.F;
    int _d_i = 0;
    int _d_j = 0;
    int _d_k = 0;
    int _d_eq_type = 0;
    int _d_xN = 0;
    int _d_yN = 0;
    int _d_zN = 0;
    float _d_dx = 0.F;
    float _d_dy = 0.F;
    float _d_dz = 0.F;
    bool _cond0;
    float _d_u_ip1 = 0.F;
    float u_ip1 = 0.F;
    float _d_u_im1 = 0.F;
    float u_im1 = 0.F;
    float _d_u_jp1 = 0.F;
    float u_jp1 = 0.F;
    float _d_u_jm1 = 0.F;
    float u_jm1 = 0.F;
    float _d_u_kp1 = 0.F;
    float u_kp1 = 0.F;
    float _d_u_km1 = 0.F;
    float u_km1 = 0.F;
    float _d_u_ijk = 0.F;
    float u_ijk = 0.F;
    float _d_v_jp1 = 0.F;
    float v_jp1 = 0.F;
    float _d_v_jm1 = 0.F;
    float v_jm1 = 0.F;
    float _d_w_kp1 = 0.F;
    float w_kp1 = 0.F;
    float _d_w_km1 = 0.F;
    float w_km1 = 0.F;
    float _d_p_ip1 = 0.F;
    float p_ip1 = 0.F;
    float _d_p_ijk = 0.F;
    float p_ijk = 0.F;
    float _d_conv_x = 0.F;
    float conv_x = 0.F;
    float _d_conv_y = 0.F;
    float conv_y = 0.F;
    float _d_conv_z = 0.F;
    float conv_z = 0.F;
    float _d_pres = 0.F;
    float pres = 0.F;
    float _d_diff = 0.F;
    float diff = 0.F;
    bool _cond1;
    float _d_v_ip1 = 0.F;
    float v_ip1 = 0.F;
    float _d_v_im1 = 0.F;
    float v_im1 = 0.F;
    float _d_v_jp10 = 0.F;
    float v_jp10 = 0.F;
    float _d_v_jm10 = 0.F;
    float v_jm10 = 0.F;
    float _d_v_kp1 = 0.F;
    float v_kp1 = 0.F;
    float _d_v_km1 = 0.F;
    float v_km1 = 0.F;
    float _d_v_ijk = 0.F;
    float v_ijk = 0.F;
    float _d_u_ip10 = 0.F;
    float u_ip10 = 0.F;
    float _d_u_im10 = 0.F;
    float u_im10 = 0.F;
    float _d_w_kp10 = 0.F;
    float w_kp10 = 0.F;
    float _d_w_km10 = 0.F;
    float w_km10 = 0.F;
    float _d_p_jp1 = 0.F;
    float p_jp1 = 0.F;
    float _d_p_ijk0 = 0.F;
    float p_ijk0 = 0.F;
    float _d_conv_x0 = 0.F;
    float conv_x0 = 0.F;
    float _d_conv_y0 = 0.F;
    float conv_y0 = 0.F;
    float _d_conv_z0 = 0.F;
    float conv_z0 = 0.F;
    float _d_pres0 = 0.F;
    float pres0 = 0.F;
    float _d_diff0 = 0.F;
    float diff0 = 0.F;
    bool _cond2;
    float _d_w_ip1 = 0.F;
    float w_ip1 = 0.F;
    float _d_w_im1 = 0.F;
    float w_im1 = 0.F;
    float _d_w_jp1 = 0.F;
    float w_jp1 = 0.F;
    float _d_w_jm1 = 0.F;
    float w_jm1 = 0.F;
    float _d_w_kp11 = 0.F;
    float w_kp11 = 0.F;
    float _d_w_km11 = 0.F;
    float w_km11 = 0.F;
    float _d_w_ijk = 0.F;
    float w_ijk = 0.F;
    float _d_u_ip11 = 0.F;
    float u_ip11 = 0.F;
    float _d_u_im11 = 0.F;
    float u_im11 = 0.F;
    float _d_v_jp11 = 0.F;
    float v_jp11 = 0.F;
    float _d_v_jm11 = 0.F;
    float v_jm11 = 0.F;
    float _d_p_kp1 = 0.F;
    float p_kp1 = 0.F;
    float _d_p_ijk1 = 0.F;
    float p_ijk1 = 0.F;
    float _d_conv_x1 = 0.F;
    float conv_x1 = 0.F;
    float _d_conv_y1 = 0.F;
    float conv_y1 = 0.F;
    float _d_conv_z1 = 0.F;
    float conv_z1 = 0.F;
    float _d_pres1 = 0.F;
    float pres1 = 0.F;
    float _d_diff1 = 0.F;
    float diff1 = 0.F;
    bool _cond3;
    float _d_u_ip12 = 0.F;
    float u_ip12 = 0.F;
    float _d_u_im12 = 0.F;
    float u_im12 = 0.F;
    float _d_v_jp12 = 0.F;
    float v_jp12 = 0.F;
    float _d_v_jm12 = 0.F;
    float v_jm12 = 0.F;
    float _d_w_kp12 = 0.F;
    float w_kp12 = 0.F;
    float _d_w_km12 = 0.F;
    float w_km12 = 0.F;
    float _d_cont = 0.F;
    float cont = 0.F;
    float _d_result = 0.F;
    float result = float(0.);
    {
        _cond0 = eq_type == 0;
        if (_cond0) {
            u_ip1 = get_field_value(y, 0, i + 1, j, k, xN, yN, zN, u_inlet);
            u_im1 = get_field_value(y, 0, i - 1, j, k, xN, yN, zN, u_inlet);
            u_jp1 = get_field_value(y, 0, i, j + 1, k, xN, yN, zN, u_inlet);
            u_jm1 = get_field_value(y, 0, i, j - 1, k, xN, yN, zN, u_inlet);
            u_kp1 = get_field_value(y, 0, i, j, k + 1, xN, yN, zN, u_inlet);
            u_km1 = get_field_value(y, 0, i, j, k - 1, xN, yN, zN, u_inlet);
            u_ijk = get_field_value(y, 0, i, j, k, xN, yN, zN, u_inlet);
            v_jp1 = get_field_value(y, 1, i, j + 1, k, xN, yN, zN, u_inlet);
            v_jm1 = get_field_value(y, 1, i, j - 1, k, xN, yN, zN, u_inlet);
            w_kp1 = get_field_value(y, 2, i, j, k + 1, xN, yN, zN, u_inlet);
            w_km1 = get_field_value(y, 2, i, j, k - 1, xN, yN, zN, u_inlet);
            p_ip1 = get_field_value(y, 3, i + 1, j, k, xN, yN, zN, u_inlet);
            p_ijk = get_field_value(y, 3, i, j, k, xN, yN, zN, u_inlet);
            conv_x = 0.5 * dy * dz * (u_ip1 * u_ip1 - u_im1 * u_im1);
            conv_y = 0.5 * dx * dz * (u_jp1 * v_jp1 - u_jm1 * v_jm1);
            conv_z = 0.5 * dx * dy * (u_kp1 * w_kp1 - u_km1 * w_km1);
            pres = (dy * dz) * (p_ip1 - p_ijk);
            diff = (1. / Re) * ((dy * dz / dx) * (u_ip1 - 2. * u_ijk + u_im1) + (dx * dz / dy) * (u_jp1 - 2. * u_ijk + u_jm1) + (dx * dy / dz) * (u_kp1 - 2. * u_ijk + u_km1));
            result = conv_x + conv_y + conv_z + pres - diff;
        } else {
            _cond1 = eq_type == 1;
            if (_cond1) {
                v_ip1 = get_field_value(y, 1, i + 1, j, k, xN, yN, zN, u_inlet);
                v_im1 = get_field_value(y, 1, i - 1, j, k, xN, yN, zN, u_inlet);
                v_jp10 = get_field_value(y, 1, i, j + 1, k, xN, yN, zN, u_inlet);
                v_jm10 = get_field_value(y, 1, i, j - 1, k, xN, yN, zN, u_inlet);
                v_kp1 = get_field_value(y, 1, i, j, k + 1, xN, yN, zN, u_inlet);
                v_km1 = get_field_value(y, 1, i, j, k - 1, xN, yN, zN, u_inlet);
                v_ijk = get_field_value(y, 1, i, j, k, xN, yN, zN, u_inlet);
                u_ip10 = get_field_value(y, 0, i + 1, j, k, xN, yN, zN, u_inlet);
                u_im10 = get_field_value(y, 0, i - 1, j, k, xN, yN, zN, u_inlet);
                w_kp10 = get_field_value(y, 2, i, j, k + 1, xN, yN, zN, u_inlet);
                w_km10 = get_field_value(y, 2, i, j, k - 1, xN, yN, zN, u_inlet);
                p_jp1 = get_field_value(y, 3, i, j + 1, k, xN, yN, zN, u_inlet);
                p_ijk0 = get_field_value(y, 3, i, j, k, xN, yN, zN, u_inlet);
                conv_x0 = 0.5 * dy * dz * (u_ip10 * v_ip1 - u_im10 * v_im1);
                conv_y0 = 0.5 * dx * dz * (v_jp10 * v_jp10 - v_jm10 * v_jm10);
                conv_z0 = 0.5 * dx * dy * (v_kp1 * w_kp10 - v_km1 * w_km10);
                pres0 = (dx * dz) * (p_jp1 - p_ijk0);
                diff0 = (1. / Re) * ((dy * dz / dx) * (v_ip1 - 2. * v_ijk + v_im1) + (dx * dz / dy) * (v_jp10 - 2. * v_ijk + v_jm10) + (dx * dy / dz) * (v_kp1 - 2. * v_ijk + v_km1));
                result = conv_x0 + conv_y0 + conv_z0 + pres0 - diff0;
            } else {
                _cond2 = eq_type == 2;
                if (_cond2) {
                    w_ip1 = get_field_value(y, 2, i + 1, j, k, xN, yN, zN, u_inlet);
                    w_im1 = get_field_value(y, 2, i - 1, j, k, xN, yN, zN, u_inlet);
                    w_jp1 = get_field_value(y, 2, i, j + 1, k, xN, yN, zN, u_inlet);
                    w_jm1 = get_field_value(y, 2, i, j - 1, k, xN, yN, zN, u_inlet);
                    w_kp11 = get_field_value(y, 2, i, j, k + 1, xN, yN, zN, u_inlet);
                    w_km11 = get_field_value(y, 2, i, j, k - 1, xN, yN, zN, u_inlet);
                    w_ijk = get_field_value(y, 2, i, j, k, xN, yN, zN, u_inlet);
                    u_ip11 = get_field_value(y, 0, i + 1, j, k, xN, yN, zN, u_inlet);
                    u_im11 = get_field_value(y, 0, i - 1, j, k, xN, yN, zN, u_inlet);
                    v_jp11 = get_field_value(y, 1, i, j + 1, k, xN, yN, zN, u_inlet);
                    v_jm11 = get_field_value(y, 1, i, j - 1, k, xN, yN, zN, u_inlet);
                    p_kp1 = get_field_value(y, 3, i, j, k + 1, xN, yN, zN, u_inlet);
                    p_ijk1 = get_field_value(y, 3, i, j, k, xN, yN, zN, u_inlet);
                    conv_x1 = 0.5 * dy * dz * (u_ip11 * w_ip1 - u_im11 * w_im1);
                    conv_y1 = 0.5 * dx * dz * (v_jp11 * w_jp1 - v_jm11 * w_jm1);
                    conv_z1 = 0.5 * dx * dy * (w_kp11 * w_kp11 - w_km11 * w_km11);
                    pres1 = (dx * dy) * (p_kp1 - p_ijk1);
                    diff1 = (1. / Re) * ((dy * dz / dx) * (w_ip1 - 2. * w_ijk + w_im1) + (dx * dz / dy) * (w_jp1 - 2. * w_ijk + w_jm1) + (dx * dy / dz) * (w_kp11 - 2. * w_ijk + w_km11));
                    result = conv_x1 + conv_y1 + conv_z1 + pres1 - diff1;
                } else {
                    _cond3 = eq_type == 3;
                    if (_cond3) {
                        u_ip12 = get_field_value(y, 0, i + 1, j, k, xN, yN, zN, u_inlet);
                        u_im12 = get_field_value(y, 0, i - 1, j, k, xN, yN, zN, u_inlet);
                        v_jp12 = get_field_value(y, 1, i, j + 1, k, xN, yN, zN, u_inlet);
                        v_jm12 = get_field_value(y, 1, i, j - 1, k, xN, yN, zN, u_inlet);
                        w_kp12 = get_field_value(y, 2, i, j, k + 1, xN, yN, zN, u_inlet);
                        w_km12 = get_field_value(y, 2, i, j, k - 1, xN, yN, zN, u_inlet);
                        cont = (dy * dz / 2.) * (u_ip12 - u_im12) + (dx * dz / 2.) * (v_jp12 - v_jm12) + (dx * dy / 2.) * (w_kp12 - w_km12);
                        result = cont;
                    }
                }
            }
        }
    }
    _d_result += 1;
    if (_cond0) {
        {
            _d_conv_x += _d_result;
            _d_conv_y += _d_result;
            _d_conv_z += _d_result;
            _d_pres += _d_result;
            _d_diff += -_d_result;
            _d_result = 0.F;
        }
        {
            double _r91 = _d_diff * ((dy * dz / dx) * (u_ip1 - 2. * u_ijk + u_im1) + (dx * dz / dy) * (u_jp1 - 2. * u_ijk + u_jm1) + (dx * dy / dz) * (u_kp1 - 2. * u_ijk + u_km1)) * -(1. / (Re * Re));
            _d_Re += _r91;
            _d_dy += (1. / Re) * _d_diff * (u_ip1 - 2. * u_ijk + u_im1) / dx * dz;
            _d_dz += dy * (1. / Re) * _d_diff * (u_ip1 - 2. * u_ijk + u_im1) / dx;
            double _r92 = (1. / Re) * _d_diff * (u_ip1 - 2. * u_ijk + u_im1) * -(dy * dz / (dx * dx));
            _d_dx += _r92;
            _d_u_ip1 += (dy * dz / dx) * (1. / Re) * _d_diff;
            _d_u_ijk += 2. * -(dy * dz / dx) * (1. / Re) * _d_diff;
            _d_u_im1 += (dy * dz / dx) * (1. / Re) * _d_diff;
            _d_dx += (1. / Re) * _d_diff * (u_jp1 - 2. * u_ijk + u_jm1) / dy * dz;
            _d_dz += dx * (1. / Re) * _d_diff * (u_jp1 - 2. * u_ijk + u_jm1) / dy;
            double _r93 = (1. / Re) * _d_diff * (u_jp1 - 2. * u_ijk + u_jm1) * -(dx * dz / (dy * dy));
            _d_dy += _r93;
            _d_u_jp1 += (dx * dz / dy) * (1. / Re) * _d_diff;
            _d_u_ijk += 2. * -(dx * dz / dy) * (1. / Re) * _d_diff;
            _d_u_jm1 += (dx * dz / dy) * (1. / Re) * _d_diff;
            _d_dx += (1. / Re) * _d_diff * (u_kp1 - 2. * u_ijk + u_km1) / dz * dy;
            _d_dy += dx * (1. / Re) * _d_diff * (u_kp1 - 2. * u_ijk + u_km1) / dz;
            double _r94 = (1. / Re) * _d_diff * (u_kp1 - 2. * u_ijk + u_km1) * -(dx * dy / (dz * dz));
            _d_dz += _r94;
            _d_u_kp1 += (dx * dy / dz) * (1. / Re) * _d_diff;
            _d_u_ijk += 2. * -(dx * dy / dz) * (1. / Re) * _d_diff;
            _d_u_km1 += (dx * dy / dz) * (1. / Re) * _d_diff;
        }
        {
            _d_dy += _d_pres * (p_ip1 - p_ijk) * dz;
            _d_dz += dy * _d_pres * (p_ip1 - p_ijk);
            _d_p_ip1 += (dy * dz) * _d_pres;
            _d_p_ijk += -(dy * dz) * _d_pres;
        }
        {
            _d_dx += 0.5 * _d_conv_z * (u_kp1 * w_kp1 - u_km1 * w_km1) * dy;
            _d_dy += 0.5 * dx * _d_conv_z * (u_kp1 * w_kp1 - u_km1 * w_km1);
            _d_u_kp1 += 0.5 * dx * dy * _d_conv_z * w_kp1;
            _d_w_kp1 += u_kp1 * 0.5 * dx * dy * _d_conv_z;
            _d_u_km1 += -0.5 * dx * dy * _d_conv_z * w_km1;
            _d_w_km1 += u_km1 * -0.5 * dx * dy * _d_conv_z;
        }
        {
            _d_dx += 0.5 * _d_conv_y * (u_jp1 * v_jp1 - u_jm1 * v_jm1) * dz;
            _d_dz += 0.5 * dx * _d_conv_y * (u_jp1 * v_jp1 - u_jm1 * v_jm1);
            _d_u_jp1 += 0.5 * dx * dz * _d_conv_y * v_jp1;
            _d_v_jp1 += u_jp1 * 0.5 * dx * dz * _d_conv_y;
            _d_u_jm1 += -0.5 * dx * dz * _d_conv_y * v_jm1;
            _d_v_jm1 += u_jm1 * -0.5 * dx * dz * _d_conv_y;
        }
        {
            _d_dy += 0.5 * _d_conv_x * (u_ip1 * u_ip1 - u_im1 * u_im1) * dz;
            _d_dz += 0.5 * dy * _d_conv_x * (u_ip1 * u_ip1 - u_im1 * u_im1);
            _d_u_ip1 += 0.5 * dy * dz * _d_conv_x * u_ip1;
            _d_u_ip1 += u_ip1 * 0.5 * dy * dz * _d_conv_x;
            _d_u_im1 += -0.5 * dy * dz * _d_conv_x * u_im1;
            _d_u_im1 += u_im1 * -0.5 * dy * dz * _d_conv_x;
        }
        {
            int _r84 = 0;
            int _r85 = 0;
            int _r86 = 0;
            int _r87 = 0;
            int _r88 = 0;
            int _r89 = 0;
            int _r90 = 0;
            get_field_value_pullback(y, 3, i, j, k, xN, yN, zN, u_inlet, _d_p_ijk, _d_y, &_r84, &_r85, &_r86, &_r87, &_r88, &_r89, &_r90);
            _d_i += _r85;
            _d_j += _r86;
            _d_k += _r87;
            _d_xN += _r88;
            _d_yN += _r89;
            _d_zN += _r90;
        }
        {
            int _r77 = 0;
            int _r78 = 0;
            int _r79 = 0;
            int _r80 = 0;
            int _r81 = 0;
            int _r82 = 0;
            int _r83 = 0;
            get_field_value_pullback(y, 3, i + 1, j, k, xN, yN, zN, u_inlet, _d_p_ip1, _d_y, &_r77, &_r78, &_r79, &_r80, &_r81, &_r82, &_r83);
            _d_i += _r78;
            _d_j += _r79;
            _d_k += _r80;
            _d_xN += _r81;
            _d_yN += _r82;
            _d_zN += _r83;
        }
        {
            int _r70 = 0;
            int _r71 = 0;
            int _r72 = 0;
            int _r73 = 0;
            int _r74 = 0;
            int _r75 = 0;
            int _r76 = 0;
            get_field_value_pullback(y, 2, i, j, k - 1, xN, yN, zN, u_inlet, _d_w_km1, _d_y, &_r70, &_r71, &_r72, &_r73, &_r74, &_r75, &_r76);
            _d_i += _r71;
            _d_j += _r72;
            _d_k += _r73;
            _d_xN += _r74;
            _d_yN += _r75;
            _d_zN += _r76;
        }
        {
            int _r63 = 0;
            int _r64 = 0;
            int _r65 = 0;
            int _r66 = 0;
            int _r67 = 0;
            int _r68 = 0;
            int _r69 = 0;
            get_field_value_pullback(y, 2, i, j, k + 1, xN, yN, zN, u_inlet, _d_w_kp1, _d_y, &_r63, &_r64, &_r65, &_r66, &_r67, &_r68, &_r69);
            _d_i += _r64;
            _d_j += _r65;
            _d_k += _r66;
            _d_xN += _r67;
            _d_yN += _r68;
            _d_zN += _r69;
        }
        {
            int _r56 = 0;
            int _r57 = 0;
            int _r58 = 0;
            int _r59 = 0;
            int _r60 = 0;
            int _r61 = 0;
            int _r62 = 0;
            get_field_value_pullback(y, 1, i, j - 1, k, xN, yN, zN, u_inlet, _d_v_jm1, _d_y, &_r56, &_r57, &_r58, &_r59, &_r60, &_r61, &_r62);
            _d_i += _r57;
            _d_j += _r58;
            _d_k += _r59;
            _d_xN += _r60;
            _d_yN += _r61;
            _d_zN += _r62;
        }
        {
            int _r49 = 0;
            int _r50 = 0;
            int _r51 = 0;
            int _r52 = 0;
            int _r53 = 0;
            int _r54 = 0;
            int _r55 = 0;
            get_field_value_pullback(y, 1, i, j + 1, k, xN, yN, zN, u_inlet, _d_v_jp1, _d_y, &_r49, &_r50, &_r51, &_r52, &_r53, &_r54, &_r55);
            _d_i += _r50;
            _d_j += _r51;
            _d_k += _r52;
            _d_xN += _r53;
            _d_yN += _r54;
            _d_zN += _r55;
        }
        {
            int _r42 = 0;
            int _r43 = 0;
            int _r44 = 0;
            int _r45 = 0;
            int _r46 = 0;
            int _r47 = 0;
            int _r48 = 0;
            get_field_value_pullback(y, 0, i, j, k, xN, yN, zN, u_inlet, _d_u_ijk, _d_y, &_r42, &_r43, &_r44, &_r45, &_r46, &_r47, &_r48);
            _d_i += _r43;
            _d_j += _r44;
            _d_k += _r45;
            _d_xN += _r46;
            _d_yN += _r47;
            _d_zN += _r48;
        }
        {
            int _r35 = 0;
            int _r36 = 0;
            int _r37 = 0;
            int _r38 = 0;
            int _r39 = 0;
            int _r40 = 0;
            int _r41 = 0;
            get_field_value_pullback(y, 0, i, j, k - 1, xN, yN, zN, u_inlet, _d_u_km1, _d_y, &_r35, &_r36, &_r37, &_r38, &_r39, &_r40, &_r41);
            _d_i += _r36;
            _d_j += _r37;
            _d_k += _r38;
            _d_xN += _r39;
            _d_yN += _r40;
            _d_zN += _r41;
        }
        {
            int _r28 = 0;
            int _r29 = 0;
            int _r30 = 0;
            int _r31 = 0;
            int _r32 = 0;
            int _r33 = 0;
            int _r34 = 0;
            get_field_value_pullback(y, 0, i, j, k + 1, xN, yN, zN, u_inlet, _d_u_kp1, _d_y, &_r28, &_r29, &_r30, &_r31, &_r32, &_r33, &_r34);
            _d_i += _r29;
            _d_j += _r30;
            _d_k += _r31;
            _d_xN += _r32;
            _d_yN += _r33;
            _d_zN += _r34;
        }
        {
            int _r21 = 0;
            int _r22 = 0;
            int _r23 = 0;
            int _r24 = 0;
            int _r25 = 0;
            int _r26 = 0;
            int _r27 = 0;
            get_field_value_pullback(y, 0, i, j - 1, k, xN, yN, zN, u_inlet, _d_u_jm1, _d_y, &_r21, &_r22, &_r23, &_r24, &_r25, &_r26, &_r27);
            _d_i += _r22;
            _d_j += _r23;
            _d_k += _r24;
            _d_xN += _r25;
            _d_yN += _r26;
            _d_zN += _r27;
        }
        {
            int _r14 = 0;
            int _r15 = 0;
            int _r16 = 0;
            int _r17 = 0;
            int _r18 = 0;
            int _r19 = 0;
            int _r20 = 0;
            get_field_value_pullback(y, 0, i, j + 1, k, xN, yN, zN, u_inlet, _d_u_jp1, _d_y, &_r14, &_r15, &_r16, &_r17, &_r18, &_r19, &_r20);
            _d_i += _r15;
            _d_j += _r16;
            _d_k += _r17;
            _d_xN += _r18;
            _d_yN += _r19;
            _d_zN += _r20;
        }
        {
            int _r7 = 0;
            int _r8 = 0;
            int _r9 = 0;
            int _r10 = 0;
            int _r11 = 0;
            int _r12 = 0;
            int _r13 = 0;
            get_field_value_pullback(y, 0, i - 1, j, k, xN, yN, zN, u_inlet, _d_u_im1, _d_y, &_r7, &_r8, &_r9, &_r10, &_r11, &_r12, &_r13);
            _d_i += _r8;
            _d_j += _r9;
            _d_k += _r10;
            _d_xN += _r11;
            _d_yN += _r12;
            _d_zN += _r13;
        }
        {
            int _r0 = 0;
            int _r1 = 0;
            int _r2 = 0;
            int _r3 = 0;
            int _r4 = 0;
            int _r5 = 0;
            int _r6 = 0;
            get_field_value_pullback(y, 0, i + 1, j, k, xN, yN, zN, u_inlet, _d_u_ip1, _d_y, &_r0, &_r1, &_r2, &_r3, &_r4, &_r5, &_r6);
            _d_i += _r1;
            _d_j += _r2;
            _d_k += _r3;
            _d_xN += _r4;
            _d_yN += _r5;
            _d_zN += _r6;
        }
    } else if (_cond1) {
        {
            _d_conv_x0 += _d_result;
            _d_conv_y0 += _d_result;
            _d_conv_z0 += _d_result;
            _d_pres0 += _d_result;
            _d_diff0 += -_d_result;
            _d_result = 0.F;
        }
        {
            double _r186 = _d_diff0 * ((dy * dz / dx) * (v_ip1 - 2. * v_ijk + v_im1) + (dx * dz / dy) * (v_jp10 - 2. * v_ijk + v_jm10) + (dx * dy / dz) * (v_kp1 - 2. * v_ijk + v_km1)) * -(1. / (Re * Re));
            _d_Re += _r186;
            _d_dy += (1. / Re) * _d_diff0 * (v_ip1 - 2. * v_ijk + v_im1) / dx * dz;
            _d_dz += dy * (1. / Re) * _d_diff0 * (v_ip1 - 2. * v_ijk + v_im1) / dx;
            double _r187 = (1. / Re) * _d_diff0 * (v_ip1 - 2. * v_ijk + v_im1) * -(dy * dz / (dx * dx));
            _d_dx += _r187;
            _d_v_ip1 += (dy * dz / dx) * (1. / Re) * _d_diff0;
            _d_v_ijk += 2. * -(dy * dz / dx) * (1. / Re) * _d_diff0;
            _d_v_im1 += (dy * dz / dx) * (1. / Re) * _d_diff0;
            _d_dx += (1. / Re) * _d_diff0 * (v_jp10 - 2. * v_ijk + v_jm10) / dy * dz;
            _d_dz += dx * (1. / Re) * _d_diff0 * (v_jp10 - 2. * v_ijk + v_jm10) / dy;
            double _r188 = (1. / Re) * _d_diff0 * (v_jp10 - 2. * v_ijk + v_jm10) * -(dx * dz / (dy * dy));
            _d_dy += _r188;
            _d_v_jp10 += (dx * dz / dy) * (1. / Re) * _d_diff0;
            _d_v_ijk += 2. * -(dx * dz / dy) * (1. / Re) * _d_diff0;
            _d_v_jm10 += (dx * dz / dy) * (1. / Re) * _d_diff0;
            _d_dx += (1. / Re) * _d_diff0 * (v_kp1 - 2. * v_ijk + v_km1) / dz * dy;
            _d_dy += dx * (1. / Re) * _d_diff0 * (v_kp1 - 2. * v_ijk + v_km1) / dz;
            double _r189 = (1. / Re) * _d_diff0 * (v_kp1 - 2. * v_ijk + v_km1) * -(dx * dy / (dz * dz));
            _d_dz += _r189;
            _d_v_kp1 += (dx * dy / dz) * (1. / Re) * _d_diff0;
            _d_v_ijk += 2. * -(dx * dy / dz) * (1. / Re) * _d_diff0;
            _d_v_km1 += (dx * dy / dz) * (1. / Re) * _d_diff0;
        }
        {
            _d_dx += _d_pres0 * (p_jp1 - p_ijk0) * dz;
            _d_dz += dx * _d_pres0 * (p_jp1 - p_ijk0);
            _d_p_jp1 += (dx * dz) * _d_pres0;
            _d_p_ijk0 += -(dx * dz) * _d_pres0;
        }
        {
            _d_dx += 0.5 * _d_conv_z0 * (v_kp1 * w_kp10 - v_km1 * w_km10) * dy;
            _d_dy += 0.5 * dx * _d_conv_z0 * (v_kp1 * w_kp10 - v_km1 * w_km10);
            _d_v_kp1 += 0.5 * dx * dy * _d_conv_z0 * w_kp10;
            _d_w_kp10 += v_kp1 * 0.5 * dx * dy * _d_conv_z0;
            _d_v_km1 += -0.5 * dx * dy * _d_conv_z0 * w_km10;
            _d_w_km10 += v_km1 * -0.5 * dx * dy * _d_conv_z0;
        }
        {
            _d_dx += 0.5 * _d_conv_y0 * (v_jp10 * v_jp10 - v_jm10 * v_jm10) * dz;
            _d_dz += 0.5 * dx * _d_conv_y0 * (v_jp10 * v_jp10 - v_jm10 * v_jm10);
            _d_v_jp10 += 0.5 * dx * dz * _d_conv_y0 * v_jp10;
            _d_v_jp10 += v_jp10 * 0.5 * dx * dz * _d_conv_y0;
            _d_v_jm10 += -0.5 * dx * dz * _d_conv_y0 * v_jm10;
            _d_v_jm10 += v_jm10 * -0.5 * dx * dz * _d_conv_y0;
        }
        {
            _d_dy += 0.5 * _d_conv_x0 * (u_ip10 * v_ip1 - u_im10 * v_im1) * dz;
            _d_dz += 0.5 * dy * _d_conv_x0 * (u_ip10 * v_ip1 - u_im10 * v_im1);
            _d_u_ip10 += 0.5 * dy * dz * _d_conv_x0 * v_ip1;
            _d_v_ip1 += u_ip10 * 0.5 * dy * dz * _d_conv_x0;
            _d_u_im10 += -0.5 * dy * dz * _d_conv_x0 * v_im1;
            _d_v_im1 += u_im10 * -0.5 * dy * dz * _d_conv_x0;
        }
        {
            int _r179 = 0;
            int _r180 = 0;
            int _r181 = 0;
            int _r182 = 0;
            int _r183 = 0;
            int _r184 = 0;
            int _r185 = 0;
            get_field_value_pullback(y, 3, i, j, k, xN, yN, zN, u_inlet, _d_p_ijk0, _d_y, &_r179, &_r180, &_r181, &_r182, &_r183, &_r184, &_r185);
            _d_i += _r180;
            _d_j += _r181;
            _d_k += _r182;
            _d_xN += _r183;
            _d_yN += _r184;
            _d_zN += _r185;
        }
        {
            int _r172 = 0;
            int _r173 = 0;
            int _r174 = 0;
            int _r175 = 0;
            int _r176 = 0;
            int _r177 = 0;
            int _r178 = 0;
            get_field_value_pullback(y, 3, i, j + 1, k, xN, yN, zN, u_inlet, _d_p_jp1, _d_y, &_r172, &_r173, &_r174, &_r175, &_r176, &_r177, &_r178);
            _d_i += _r173;
            _d_j += _r174;
            _d_k += _r175;
            _d_xN += _r176;
            _d_yN += _r177;
            _d_zN += _r178;
        }
        {
            int _r165 = 0;
            int _r166 = 0;
            int _r167 = 0;
            int _r168 = 0;
            int _r169 = 0;
            int _r170 = 0;
            int _r171 = 0;
            get_field_value_pullback(y, 2, i, j, k - 1, xN, yN, zN, u_inlet, _d_w_km10, _d_y, &_r165, &_r166, &_r167, &_r168, &_r169, &_r170, &_r171);
            _d_i += _r166;
            _d_j += _r167;
            _d_k += _r168;
            _d_xN += _r169;
            _d_yN += _r170;
            _d_zN += _r171;
        }
        {
            int _r158 = 0;
            int _r159 = 0;
            int _r160 = 0;
            int _r161 = 0;
            int _r162 = 0;
            int _r163 = 0;
            int _r164 = 0;
            get_field_value_pullback(y, 2, i, j, k + 1, xN, yN, zN, u_inlet, _d_w_kp10, _d_y, &_r158, &_r159, &_r160, &_r161, &_r162, &_r163, &_r164);
            _d_i += _r159;
            _d_j += _r160;
            _d_k += _r161;
            _d_xN += _r162;
            _d_yN += _r163;
            _d_zN += _r164;
        }
        {
            int _r151 = 0;
            int _r152 = 0;
            int _r153 = 0;
            int _r154 = 0;
            int _r155 = 0;
            int _r156 = 0;
            int _r157 = 0;
            get_field_value_pullback(y, 0, i - 1, j, k, xN, yN, zN, u_inlet, _d_u_im10, _d_y, &_r151, &_r152, &_r153, &_r154, &_r155, &_r156, &_r157);
            _d_i += _r152;
            _d_j += _r153;
            _d_k += _r154;
            _d_xN += _r155;
            _d_yN += _r156;
            _d_zN += _r157;
        }
        {
            int _r144 = 0;
            int _r145 = 0;
            int _r146 = 0;
            int _r147 = 0;
            int _r148 = 0;
            int _r149 = 0;
            int _r150 = 0;
            get_field_value_pullback(y, 0, i + 1, j, k, xN, yN, zN, u_inlet, _d_u_ip10, _d_y, &_r144, &_r145, &_r146, &_r147, &_r148, &_r149, &_r150);
            _d_i += _r145;
            _d_j += _r146;
            _d_k += _r147;
            _d_xN += _r148;
            _d_yN += _r149;
            _d_zN += _r150;
        }
        {
            int _r137 = 0;
            int _r138 = 0;
            int _r139 = 0;
            int _r140 = 0;
            int _r141 = 0;
            int _r142 = 0;
            int _r143 = 0;
            get_field_value_pullback(y, 1, i, j, k, xN, yN, zN, u_inlet, _d_v_ijk, _d_y, &_r137, &_r138, &_r139, &_r140, &_r141, &_r142, &_r143);
            _d_i += _r138;
            _d_j += _r139;
            _d_k += _r140;
            _d_xN += _r141;
            _d_yN += _r142;
            _d_zN += _r143;
        }
        {
            int _r130 = 0;
            int _r131 = 0;
            int _r132 = 0;
            int _r133 = 0;
            int _r134 = 0;
            int _r135 = 0;
            int _r136 = 0;
            get_field_value_pullback(y, 1, i, j, k - 1, xN, yN, zN, u_inlet, _d_v_km1, _d_y, &_r130, &_r131, &_r132, &_r133, &_r134, &_r135, &_r136);
            _d_i += _r131;
            _d_j += _r132;
            _d_k += _r133;
            _d_xN += _r134;
            _d_yN += _r135;
            _d_zN += _r136;
        }
        {
            int _r123 = 0;
            int _r124 = 0;
            int _r125 = 0;
            int _r126 = 0;
            int _r127 = 0;
            int _r128 = 0;
            int _r129 = 0;
            get_field_value_pullback(y, 1, i, j, k + 1, xN, yN, zN, u_inlet, _d_v_kp1, _d_y, &_r123, &_r124, &_r125, &_r126, &_r127, &_r128, &_r129);
            _d_i += _r124;
            _d_j += _r125;
            _d_k += _r126;
            _d_xN += _r127;
            _d_yN += _r128;
            _d_zN += _r129;
        }
        {
            int _r116 = 0;
            int _r117 = 0;
            int _r118 = 0;
            int _r119 = 0;
            int _r120 = 0;
            int _r121 = 0;
            int _r122 = 0;
            get_field_value_pullback(y, 1, i, j - 1, k, xN, yN, zN, u_inlet, _d_v_jm10, _d_y, &_r116, &_r117, &_r118, &_r119, &_r120, &_r121, &_r122);
            _d_i += _r117;
            _d_j += _r118;
            _d_k += _r119;
            _d_xN += _r120;
            _d_yN += _r121;
            _d_zN += _r122;
        }
        {
            int _r109 = 0;
            int _r110 = 0;
            int _r111 = 0;
            int _r112 = 0;
            int _r113 = 0;
            int _r114 = 0;
            int _r115 = 0;
            get_field_value_pullback(y, 1, i, j + 1, k, xN, yN, zN, u_inlet, _d_v_jp10, _d_y, &_r109, &_r110, &_r111, &_r112, &_r113, &_r114, &_r115);
            _d_i += _r110;
            _d_j += _r111;
            _d_k += _r112;
            _d_xN += _r113;
            _d_yN += _r114;
            _d_zN += _r115;
        }
        {
            int _r102 = 0;
            int _r103 = 0;
            int _r104 = 0;
            int _r105 = 0;
            int _r106 = 0;
            int _r107 = 0;
            int _r108 = 0;
            get_field_value_pullback(y, 1, i - 1, j, k, xN, yN, zN, u_inlet, _d_v_im1, _d_y, &_r102, &_r103, &_r104, &_r105, &_r106, &_r107, &_r108);
            _d_i += _r103;
            _d_j += _r104;
            _d_k += _r105;
            _d_xN += _r106;
            _d_yN += _r107;
            _d_zN += _r108;
        }
        {
            int _r95 = 0;
            int _r96 = 0;
            int _r97 = 0;
            int _r98 = 0;
            int _r99 = 0;
            int _r100 = 0;
            int _r101 = 0;
            get_field_value_pullback(y, 1, i + 1, j, k, xN, yN, zN, u_inlet, _d_v_ip1, _d_y, &_r95, &_r96, &_r97, &_r98, &_r99, &_r100, &_r101);
            _d_i += _r96;
            _d_j += _r97;
            _d_k += _r98;
            _d_xN += _r99;
            _d_yN += _r100;
            _d_zN += _r101;
        }
    } else if (_cond2) {
        {
            _d_conv_x1 += _d_result;
            _d_conv_y1 += _d_result;
            _d_conv_z1 += _d_result;
            _d_pres1 += _d_result;
            _d_diff1 += -_d_result;
            _d_result = 0.F;
        }
        {
            double _r281 = _d_diff1 * ((dy * dz / dx) * (w_ip1 - 2. * w_ijk + w_im1) + (dx * dz / dy) * (w_jp1 - 2. * w_ijk + w_jm1) + (dx * dy / dz) * (w_kp11 - 2. * w_ijk + w_km11)) * -(1. / (Re * Re));
            _d_Re += _r281;
            _d_dy += (1. / Re) * _d_diff1 * (w_ip1 - 2. * w_ijk + w_im1) / dx * dz;
            _d_dz += dy * (1. / Re) * _d_diff1 * (w_ip1 - 2. * w_ijk + w_im1) / dx;
            double _r282 = (1. / Re) * _d_diff1 * (w_ip1 - 2. * w_ijk + w_im1) * -(dy * dz / (dx * dx));
            _d_dx += _r282;
            _d_w_ip1 += (dy * dz / dx) * (1. / Re) * _d_diff1;
            _d_w_ijk += 2. * -(dy * dz / dx) * (1. / Re) * _d_diff1;
            _d_w_im1 += (dy * dz / dx) * (1. / Re) * _d_diff1;
            _d_dx += (1. / Re) * _d_diff1 * (w_jp1 - 2. * w_ijk + w_jm1) / dy * dz;
            _d_dz += dx * (1. / Re) * _d_diff1 * (w_jp1 - 2. * w_ijk + w_jm1) / dy;
            double _r283 = (1. / Re) * _d_diff1 * (w_jp1 - 2. * w_ijk + w_jm1) * -(dx * dz / (dy * dy));
            _d_dy += _r283;
            _d_w_jp1 += (dx * dz / dy) * (1. / Re) * _d_diff1;
            _d_w_ijk += 2. * -(dx * dz / dy) * (1. / Re) * _d_diff1;
            _d_w_jm1 += (dx * dz / dy) * (1. / Re) * _d_diff1;
            _d_dx += (1. / Re) * _d_diff1 * (w_kp11 - 2. * w_ijk + w_km11) / dz * dy;
            _d_dy += dx * (1. / Re) * _d_diff1 * (w_kp11 - 2. * w_ijk + w_km11) / dz;
            double _r284 = (1. / Re) * _d_diff1 * (w_kp11 - 2. * w_ijk + w_km11) * -(dx * dy / (dz * dz));
            _d_dz += _r284;
            _d_w_kp11 += (dx * dy / dz) * (1. / Re) * _d_diff1;
            _d_w_ijk += 2. * -(dx * dy / dz) * (1. / Re) * _d_diff1;
            _d_w_km11 += (dx * dy / dz) * (1. / Re) * _d_diff1;
        }
        {
            _d_dx += _d_pres1 * (p_kp1 - p_ijk1) * dy;
            _d_dy += dx * _d_pres1 * (p_kp1 - p_ijk1);
            _d_p_kp1 += (dx * dy) * _d_pres1;
            _d_p_ijk1 += -(dx * dy) * _d_pres1;
        }
        {
            _d_dx += 0.5 * _d_conv_z1 * (w_kp11 * w_kp11 - w_km11 * w_km11) * dy;
            _d_dy += 0.5 * dx * _d_conv_z1 * (w_kp11 * w_kp11 - w_km11 * w_km11);
            _d_w_kp11 += 0.5 * dx * dy * _d_conv_z1 * w_kp11;
            _d_w_kp11 += w_kp11 * 0.5 * dx * dy * _d_conv_z1;
            _d_w_km11 += -0.5 * dx * dy * _d_conv_z1 * w_km11;
            _d_w_km11 += w_km11 * -0.5 * dx * dy * _d_conv_z1;
        }
        {
            _d_dx += 0.5 * _d_conv_y1 * (v_jp11 * w_jp1 - v_jm11 * w_jm1) * dz;
            _d_dz += 0.5 * dx * _d_conv_y1 * (v_jp11 * w_jp1 - v_jm11 * w_jm1);
            _d_v_jp11 += 0.5 * dx * dz * _d_conv_y1 * w_jp1;
            _d_w_jp1 += v_jp11 * 0.5 * dx * dz * _d_conv_y1;
            _d_v_jm11 += -0.5 * dx * dz * _d_conv_y1 * w_jm1;
            _d_w_jm1 += v_jm11 * -0.5 * dx * dz * _d_conv_y1;
        }
        {
            _d_dy += 0.5 * _d_conv_x1 * (u_ip11 * w_ip1 - u_im11 * w_im1) * dz;
            _d_dz += 0.5 * dy * _d_conv_x1 * (u_ip11 * w_ip1 - u_im11 * w_im1);
            _d_u_ip11 += 0.5 * dy * dz * _d_conv_x1 * w_ip1;
            _d_w_ip1 += u_ip11 * 0.5 * dy * dz * _d_conv_x1;
            _d_u_im11 += -0.5 * dy * dz * _d_conv_x1 * w_im1;
            _d_w_im1 += u_im11 * -0.5 * dy * dz * _d_conv_x1;
        }
        {
            int _r274 = 0;
            int _r275 = 0;
            int _r276 = 0;
            int _r277 = 0;
            int _r278 = 0;
            int _r279 = 0;
            int _r280 = 0;
            get_field_value_pullback(y, 3, i, j, k, xN, yN, zN, u_inlet, _d_p_ijk1, _d_y, &_r274, &_r275, &_r276, &_r277, &_r278, &_r279, &_r280);
            _d_i += _r275;
            _d_j += _r276;
            _d_k += _r277;
            _d_xN += _r278;
            _d_yN += _r279;
            _d_zN += _r280;
        }
        {
            int _r267 = 0;
            int _r268 = 0;
            int _r269 = 0;
            int _r270 = 0;
            int _r271 = 0;
            int _r272 = 0;
            int _r273 = 0;
            get_field_value_pullback(y, 3, i, j, k + 1, xN, yN, zN, u_inlet, _d_p_kp1, _d_y, &_r267, &_r268, &_r269, &_r270, &_r271, &_r272, &_r273);
            _d_i += _r268;
            _d_j += _r269;
            _d_k += _r270;
            _d_xN += _r271;
            _d_yN += _r272;
            _d_zN += _r273;
        }
        {
            int _r260 = 0;
            int _r261 = 0;
            int _r262 = 0;
            int _r263 = 0;
            int _r264 = 0;
            int _r265 = 0;
            int _r266 = 0;
            get_field_value_pullback(y, 1, i, j - 1, k, xN, yN, zN, u_inlet, _d_v_jm11, _d_y, &_r260, &_r261, &_r262, &_r263, &_r264, &_r265, &_r266);
            _d_i += _r261;
            _d_j += _r262;
            _d_k += _r263;
            _d_xN += _r264;
            _d_yN += _r265;
            _d_zN += _r266;
        }
        {
            int _r253 = 0;
            int _r254 = 0;
            int _r255 = 0;
            int _r256 = 0;
            int _r257 = 0;
            int _r258 = 0;
            int _r259 = 0;
            get_field_value_pullback(y, 1, i, j + 1, k, xN, yN, zN, u_inlet, _d_v_jp11, _d_y, &_r253, &_r254, &_r255, &_r256, &_r257, &_r258, &_r259);
            _d_i += _r254;
            _d_j += _r255;
            _d_k += _r256;
            _d_xN += _r257;
            _d_yN += _r258;
            _d_zN += _r259;
        }
        {
            int _r246 = 0;
            int _r247 = 0;
            int _r248 = 0;
            int _r249 = 0;
            int _r250 = 0;
            int _r251 = 0;
            int _r252 = 0;
            get_field_value_pullback(y, 0, i - 1, j, k, xN, yN, zN, u_inlet, _d_u_im11, _d_y, &_r246, &_r247, &_r248, &_r249, &_r250, &_r251, &_r252);
            _d_i += _r247;
            _d_j += _r248;
            _d_k += _r249;
            _d_xN += _r250;
            _d_yN += _r251;
            _d_zN += _r252;
        }
        {
            int _r239 = 0;
            int _r240 = 0;
            int _r241 = 0;
            int _r242 = 0;
            int _r243 = 0;
            int _r244 = 0;
            int _r245 = 0;
            get_field_value_pullback(y, 0, i + 1, j, k, xN, yN, zN, u_inlet, _d_u_ip11, _d_y, &_r239, &_r240, &_r241, &_r242, &_r243, &_r244, &_r245);
            _d_i += _r240;
            _d_j += _r241;
            _d_k += _r242;
            _d_xN += _r243;
            _d_yN += _r244;
            _d_zN += _r245;
        }
        {
            int _r232 = 0;
            int _r233 = 0;
            int _r234 = 0;
            int _r235 = 0;
            int _r236 = 0;
            int _r237 = 0;
            int _r238 = 0;
            get_field_value_pullback(y, 2, i, j, k, xN, yN, zN, u_inlet, _d_w_ijk, _d_y, &_r232, &_r233, &_r234, &_r235, &_r236, &_r237, &_r238);
            _d_i += _r233;
            _d_j += _r234;
            _d_k += _r235;
            _d_xN += _r236;
            _d_yN += _r237;
            _d_zN += _r238;
        }
        {
            int _r225 = 0;
            int _r226 = 0;
            int _r227 = 0;
            int _r228 = 0;
            int _r229 = 0;
            int _r230 = 0;
            int _r231 = 0;
            get_field_value_pullback(y, 2, i, j, k - 1, xN, yN, zN, u_inlet, _d_w_km11, _d_y, &_r225, &_r226, &_r227, &_r228, &_r229, &_r230, &_r231);
            _d_i += _r226;
            _d_j += _r227;
            _d_k += _r228;
            _d_xN += _r229;
            _d_yN += _r230;
            _d_zN += _r231;
        }
        {
            int _r218 = 0;
            int _r219 = 0;
            int _r220 = 0;
            int _r221 = 0;
            int _r222 = 0;
            int _r223 = 0;
            int _r224 = 0;
            get_field_value_pullback(y, 2, i, j, k + 1, xN, yN, zN, u_inlet, _d_w_kp11, _d_y, &_r218, &_r219, &_r220, &_r221, &_r222, &_r223, &_r224);
            _d_i += _r219;
            _d_j += _r220;
            _d_k += _r221;
            _d_xN += _r222;
            _d_yN += _r223;
            _d_zN += _r224;
        }
        {
            int _r211 = 0;
            int _r212 = 0;
            int _r213 = 0;
            int _r214 = 0;
            int _r215 = 0;
            int _r216 = 0;
            int _r217 = 0;
            get_field_value_pullback(y, 2, i, j - 1, k, xN, yN, zN, u_inlet, _d_w_jm1, _d_y, &_r211, &_r212, &_r213, &_r214, &_r215, &_r216, &_r217);
            _d_i += _r212;
            _d_j += _r213;
            _d_k += _r214;
            _d_xN += _r215;
            _d_yN += _r216;
            _d_zN += _r217;
        }
        {
            int _r204 = 0;
            int _r205 = 0;
            int _r206 = 0;
            int _r207 = 0;
            int _r208 = 0;
            int _r209 = 0;
            int _r210 = 0;
            get_field_value_pullback(y, 2, i, j + 1, k, xN, yN, zN, u_inlet, _d_w_jp1, _d_y, &_r204, &_r205, &_r206, &_r207, &_r208, &_r209, &_r210);
            _d_i += _r205;
            _d_j += _r206;
            _d_k += _r207;
            _d_xN += _r208;
            _d_yN += _r209;
            _d_zN += _r210;
        }
        {
            int _r197 = 0;
            int _r198 = 0;
            int _r199 = 0;
            int _r200 = 0;
            int _r201 = 0;
            int _r202 = 0;
            int _r203 = 0;
            get_field_value_pullback(y, 2, i - 1, j, k, xN, yN, zN, u_inlet, _d_w_im1, _d_y, &_r197, &_r198, &_r199, &_r200, &_r201, &_r202, &_r203);
            _d_i += _r198;
            _d_j += _r199;
            _d_k += _r200;
            _d_xN += _r201;
            _d_yN += _r202;
            _d_zN += _r203;
        }
        {
            int _r190 = 0;
            int _r191 = 0;
            int _r192 = 0;
            int _r193 = 0;
            int _r194 = 0;
            int _r195 = 0;
            int _r196 = 0;
            get_field_value_pullback(y, 2, i + 1, j, k, xN, yN, zN, u_inlet, _d_w_ip1, _d_y, &_r190, &_r191, &_r192, &_r193, &_r194, &_r195, &_r196);
            _d_i += _r191;
            _d_j += _r192;
            _d_k += _r193;
            _d_xN += _r194;
            _d_yN += _r195;
            _d_zN += _r196;
        }
    } else if (_cond3) {
        {
            _d_cont += _d_result;
            _d_result = 0.F;
        }
        {
            _d_dy += _d_cont * (u_ip12 - u_im12) / 2. * dz;
            _d_dz += dy * _d_cont * (u_ip12 - u_im12) / 2.;
            _d_u_ip12 += (dy * dz / 2.) * _d_cont;
            _d_u_im12 += -(dy * dz / 2.) * _d_cont;
            _d_dx += _d_cont * (v_jp12 - v_jm12) / 2. * dz;
            _d_dz += dx * _d_cont * (v_jp12 - v_jm12) / 2.;
            _d_v_jp12 += (dx * dz / 2.) * _d_cont;
            _d_v_jm12 += -(dx * dz / 2.) * _d_cont;
            _d_dx += _d_cont * (w_kp12 - w_km12) / 2. * dy;
            _d_dy += dx * _d_cont * (w_kp12 - w_km12) / 2.;
            _d_w_kp12 += (dx * dy / 2.) * _d_cont;
            _d_w_km12 += -(dx * dy / 2.) * _d_cont;
        }
        {
            int _r320 = 0;
            int _r321 = 0;
            int _r322 = 0;
            int _r323 = 0;
            int _r324 = 0;
            int _r325 = 0;
            int _r326 = 0;
            get_field_value_pullback(y, 2, i, j, k - 1, xN, yN, zN, u_inlet, _d_w_km12, _d_y, &_r320, &_r321, &_r322, &_r323, &_r324, &_r325, &_r326);
            _d_i += _r321;
            _d_j += _r322;
            _d_k += _r323;
            _d_xN += _r324;
            _d_yN += _r325;
            _d_zN += _r326;
        }
        {
            int _r313 = 0;
            int _r314 = 0;
            int _r315 = 0;
            int _r316 = 0;
            int _r317 = 0;
            int _r318 = 0;
            int _r319 = 0;
            get_field_value_pullback(y, 2, i, j, k + 1, xN, yN, zN, u_inlet, _d_w_kp12, _d_y, &_r313, &_r314, &_r315, &_r316, &_r317, &_r318, &_r319);
            _d_i += _r314;
            _d_j += _r315;
            _d_k += _r316;
            _d_xN += _r317;
            _d_yN += _r318;
            _d_zN += _r319;
        }
        {
            int _r306 = 0;
            int _r307 = 0;
            int _r308 = 0;
            int _r309 = 0;
            int _r310 = 0;
            int _r311 = 0;
            int _r312 = 0;
            get_field_value_pullback(y, 1, i, j - 1, k, xN, yN, zN, u_inlet, _d_v_jm12, _d_y, &_r306, &_r307, &_r308, &_r309, &_r310, &_r311, &_r312);
            _d_i += _r307;
            _d_j += _r308;
            _d_k += _r309;
            _d_xN += _r310;
            _d_yN += _r311;
            _d_zN += _r312;
        }
        {
            int _r299 = 0;
            int _r300 = 0;
            int _r301 = 0;
            int _r302 = 0;
            int _r303 = 0;
            int _r304 = 0;
            int _r305 = 0;
            get_field_value_pullback(y, 1, i, j + 1, k, xN, yN, zN, u_inlet, _d_v_jp12, _d_y, &_r299, &_r300, &_r301, &_r302, &_r303, &_r304, &_r305);
            _d_i += _r300;
            _d_j += _r301;
            _d_k += _r302;
            _d_xN += _r303;
            _d_yN += _r304;
            _d_zN += _r305;
        }
        {
            int _r292 = 0;
            int _r293 = 0;
            int _r294 = 0;
            int _r295 = 0;
            int _r296 = 0;
            int _r297 = 0;
            int _r298 = 0;
            get_field_value_pullback(y, 0, i - 1, j, k, xN, yN, zN, u_inlet, _d_u_im12, _d_y, &_r292, &_r293, &_r294, &_r295, &_r296, &_r297, &_r298);
            _d_i += _r293;
            _d_j += _r294;
            _d_k += _r295;
            _d_xN += _r296;
            _d_yN += _r297;
            _d_zN += _r298;
        }
        {
            int _r285 = 0;
            int _r286 = 0;
            int _r287 = 0;
            int _r288 = 0;
            int _r289 = 0;
            int _r290 = 0;
            int _r291 = 0;
            get_field_value_pullback(y, 0, i + 1, j, k, xN, yN, zN, u_inlet, _d_u_ip12, _d_y, &_r285, &_r286, &_r287, &_r288, &_r289, &_r290, &_r291);
            _d_i += _r286;
            _d_j += _r287;
            _d_k += _r288;
            _d_xN += _r289;
            _d_yN += _r290;
            _d_zN += _r291;
        }
    }
}
inline void get_field_value_pullback(const float *y, int field_type, int i, int j, int k, int xN, int yN, int zN, const float *u_inlet, float _d_y0, float *_d_y, int *_d_field_type, int *_d_i, int *_d_j, int *_d_k, int *_d_xN, int *_d_yN, int *_d_zN) {
    bool _cond0;
    bool _cond1;
    bool _cond2;
    bool _cond3;
    bool _cond4;
    bool _cond5;
    bool _cond6;
    bool _cond7;
    bool _cond8;
    bool _cond9;
    bool _cond10;
    bool _cond11;
    int _d_sizeY = 0;
    const int sizeY = yN + 2;
    int _d_sizeZ = 0;
    const int sizeZ = zN + 2;
    int _d_nCell = 0;
    const int nCell = xN * yN * zN;
    int _d_i_clamped = 0;
    int i_clamped = i;
    int _d_j_clamped = 0;
    int j_clamped = j;
    int _d_k_clamped = 0;
    int k_clamped = k;
    {
        _cond0 = i < 1;
        if (_cond0) {
            i_clamped = 1;
            {
                _cond1 = field_type == 0;
                if (_cond1) {
                    goto _label0;
                }
            }
            goto _label1;
        }
    }
    {
        _cond2 = i > xN;
        if (_cond2) {
            i_clamped = xN;
            {
                _cond3 = field_type == 3;
                if (_cond3) {
                    goto _label2;
                }
            }
        }
    }
    {
        _cond4 = j < 1;
        if (_cond4) {
            j_clamped = 1;
            {
                _cond5 = field_type != 3;
                if (_cond5)
                    goto _label3;
            }
        }
    }
    {
        _cond6 = j > yN;
        if (_cond6) {
            j_clamped = yN;
            {
                _cond7 = field_type != 3;
                if (_cond7)
                    goto _label4;
            }
        }
    }
    {
        _cond8 = k < 1;
        if (_cond8) {
            k_clamped = 1;
            {
                _cond9 = field_type != 3;
                if (_cond9)
                    goto _label5;
            }
        }
    }
    {
        _cond10 = k > zN;
        if (_cond10) {
            k_clamped = zN;
            {
                _cond11 = field_type != 3;
                if (_cond11)
                    goto _label6;
            }
        }
    }
    int _d_pos = 0;
    const int pos = (i_clamped - 1) * yN * zN + (j_clamped - 1) * zN + (k_clamped - 1);
    _d_y[field_type * nCell + pos] += _d_y0;
    {
        _d_i_clamped += _d_pos * zN * yN;
        *_d_yN += (i_clamped - 1) * _d_pos * zN;
        *_d_zN += (i_clamped - 1) * yN * _d_pos;
        _d_j_clamped += _d_pos * zN;
        *_d_zN += (j_clamped - 1) * _d_pos;
        _d_k_clamped += _d_pos;
    }
    if (_cond10) {
        if (_cond11)
          _label6:
            ;
        {
            *_d_zN += _d_k_clamped;
            _d_k_clamped = 0;
        }
    }
    if (_cond8) {
        if (_cond9)
          _label5:
            ;
        _d_k_clamped = 0;
    }
    if (_cond6) {
        if (_cond7)
          _label4:
            ;
        {
            *_d_yN += _d_j_clamped;
            _d_j_clamped = 0;
        }
    }
    if (_cond4) {
        if (_cond5)
          _label3:
            ;
        _d_j_clamped = 0;
    }
    if (_cond2) {
        if (_cond3) {
          _label2:
            ;
        }
        {
            *_d_xN += _d_i_clamped;
            _d_i_clamped = 0;
        }
    }
    if (_cond0) {
      _label1:
        ;
        if (_cond1) {
          _label0:
            ;
        }
        _d_i_clamped = 0;
    }
    *_d_k += _d_k_clamped;
    *_d_j += _d_j_clamped;
    *_d_i += _d_i_clamped;
    {
        *_d_xN += _d_nCell * zN * yN;
        *_d_yN += xN * _d_nCell * zN;
        *_d_zN += xN * yN * _d_nCell;
    }
    *_d_zN += _d_sizeZ;
    *_d_yN += _d_sizeY;
}
void compute_residual_component_grad_1(float Re, const float *y, int i, int j, int k, int eq_type, int xN, int yN, int zN, float dx, float dy, float dz, const float *u_inlet, float *_d_y) {
    float _d_Re = 0.F;
    int _d_i = 0;
    int _d_j = 0;
    int _d_k = 0;
    int _d_eq_type = 0;
    int _d_xN = 0;
    int _d_yN = 0;
    int _d_zN = 0;
    float _d_dx = 0.F;
    float _d_dy = 0.F;
    float _d_dz = 0.F;
    bool _cond0;
    float _d_u_ip1 = 0.F;
    float u_ip1 = 0.F;
    float _d_u_im1 = 0.F;
    float u_im1 = 0.F;
    float _d_u_jp1 = 0.F;
    float u_jp1 = 0.F;
    float _d_u_jm1 = 0.F;
    float u_jm1 = 0.F;
    float _d_u_kp1 = 0.F;
    float u_kp1 = 0.F;
    float _d_u_km1 = 0.F;
    float u_km1 = 0.F;
    float _d_u_ijk = 0.F;
    float u_ijk = 0.F;
    float _d_v_jp1 = 0.F;
    float v_jp1 = 0.F;
    float _d_v_jm1 = 0.F;
    float v_jm1 = 0.F;
    float _d_w_kp1 = 0.F;
    float w_kp1 = 0.F;
    float _d_w_km1 = 0.F;
    float w_km1 = 0.F;
    float _d_p_ip1 = 0.F;
    float p_ip1 = 0.F;
    float _d_p_ijk = 0.F;
    float p_ijk = 0.F;
    float _d_conv_x = 0.F;
    float conv_x = 0.F;
    float _d_conv_y = 0.F;
    float conv_y = 0.F;
    float _d_conv_z = 0.F;
    float conv_z = 0.F;
    float _d_pres = 0.F;
    float pres = 0.F;
    float _d_diff = 0.F;
    float diff = 0.F;
    bool _cond1;
    float _d_v_ip1 = 0.F;
    float v_ip1 = 0.F;
    float _d_v_im1 = 0.F;
    float v_im1 = 0.F;
    float _d_v_jp10 = 0.F;
    float v_jp10 = 0.F;
    float _d_v_jm10 = 0.F;
    float v_jm10 = 0.F;
    float _d_v_kp1 = 0.F;
    float v_kp1 = 0.F;
    float _d_v_km1 = 0.F;
    float v_km1 = 0.F;
    float _d_v_ijk = 0.F;
    float v_ijk = 0.F;
    float _d_u_ip10 = 0.F;
    float u_ip10 = 0.F;
    float _d_u_im10 = 0.F;
    float u_im10 = 0.F;
    float _d_w_kp10 = 0.F;
    float w_kp10 = 0.F;
    float _d_w_km10 = 0.F;
    float w_km10 = 0.F;
    float _d_p_jp1 = 0.F;
    float p_jp1 = 0.F;
    float _d_p_ijk0 = 0.F;
    float p_ijk0 = 0.F;
    float _d_conv_x0 = 0.F;
    float conv_x0 = 0.F;
    float _d_conv_y0 = 0.F;
    float conv_y0 = 0.F;
    float _d_conv_z0 = 0.F;
    float conv_z0 = 0.F;
    float _d_pres0 = 0.F;
    float pres0 = 0.F;
    float _d_diff0 = 0.F;
    float diff0 = 0.F;
    bool _cond2;
    float _d_w_ip1 = 0.F;
    float w_ip1 = 0.F;
    float _d_w_im1 = 0.F;
    float w_im1 = 0.F;
    float _d_w_jp1 = 0.F;
    float w_jp1 = 0.F;
    float _d_w_jm1 = 0.F;
    float w_jm1 = 0.F;
    float _d_w_kp11 = 0.F;
    float w_kp11 = 0.F;
    float _d_w_km11 = 0.F;
    float w_km11 = 0.F;
    float _d_w_ijk = 0.F;
    float w_ijk = 0.F;
    float _d_u_ip11 = 0.F;
    float u_ip11 = 0.F;
    float _d_u_im11 = 0.F;
    float u_im11 = 0.F;
    float _d_v_jp11 = 0.F;
    float v_jp11 = 0.F;
    float _d_v_jm11 = 0.F;
    float v_jm11 = 0.F;
    float _d_p_kp1 = 0.F;
    float p_kp1 = 0.F;
    float _d_p_ijk1 = 0.F;
    float p_ijk1 = 0.F;
    float _d_conv_x1 = 0.F;
    float conv_x1 = 0.F;
    float _d_conv_y1 = 0.F;
    float conv_y1 = 0.F;
    float _d_conv_z1 = 0.F;
    float conv_z1 = 0.F;
    float _d_pres1 = 0.F;
    float pres1 = 0.F;
    float _d_diff1 = 0.F;
    float diff1 = 0.F;
    bool _cond3;
    float _d_u_ip12 = 0.F;
    float u_ip12 = 0.F;
    float _d_u_im12 = 0.F;
    float u_im12 = 0.F;
    float _d_v_jp12 = 0.F;
    float v_jp12 = 0.F;
    float _d_v_jm12 = 0.F;
    float v_jm12 = 0.F;
    float _d_w_kp12 = 0.F;
    float w_kp12 = 0.F;
    float _d_w_km12 = 0.F;
    float w_km12 = 0.F;
    float _d_cont = 0.F;
    float cont = 0.F;
    float _d_result = 0.F;
    float result = float(0.);
    {
        _cond0 = eq_type == 0;
        if (_cond0) {
            u_ip1 = get_field_value(y, 0, i + 1, j, k, xN, yN, zN, u_inlet);
            u_im1 = get_field_value(y, 0, i - 1, j, k, xN, yN, zN, u_inlet);
            u_jp1 = get_field_value(y, 0, i, j + 1, k, xN, yN, zN, u_inlet);
            u_jm1 = get_field_value(y, 0, i, j - 1, k, xN, yN, zN, u_inlet);
            u_kp1 = get_field_value(y, 0, i, j, k + 1, xN, yN, zN, u_inlet);
            u_km1 = get_field_value(y, 0, i, j, k - 1, xN, yN, zN, u_inlet);
            u_ijk = get_field_value(y, 0, i, j, k, xN, yN, zN, u_inlet);
            v_jp1 = get_field_value(y, 1, i, j + 1, k, xN, yN, zN, u_inlet);
            v_jm1 = get_field_value(y, 1, i, j - 1, k, xN, yN, zN, u_inlet);
            w_kp1 = get_field_value(y, 2, i, j, k + 1, xN, yN, zN, u_inlet);
            w_km1 = get_field_value(y, 2, i, j, k - 1, xN, yN, zN, u_inlet);
            p_ip1 = get_field_value(y, 3, i + 1, j, k, xN, yN, zN, u_inlet);
            p_ijk = get_field_value(y, 3, i, j, k, xN, yN, zN, u_inlet);
            conv_x = 0.5 * dy * dz * (u_ip1 * u_ip1 - u_im1 * u_im1);
            conv_y = 0.5 * dx * dz * (u_jp1 * v_jp1 - u_jm1 * v_jm1);
            conv_z = 0.5 * dx * dy * (u_kp1 * w_kp1 - u_km1 * w_km1);
            pres = (dy * dz) * (p_ip1 - p_ijk);
            diff = (1. / Re) * ((dy * dz / dx) * (u_ip1 - 2. * u_ijk + u_im1) + (dx * dz / dy) * (u_jp1 - 2. * u_ijk + u_jm1) + (dx * dy / dz) * (u_kp1 - 2. * u_ijk + u_km1));
            result = conv_x + conv_y + conv_z + pres - diff;
        } else {
            _cond1 = eq_type == 1;
            if (_cond1) {
                v_ip1 = get_field_value(y, 1, i + 1, j, k, xN, yN, zN, u_inlet);
                v_im1 = get_field_value(y, 1, i - 1, j, k, xN, yN, zN, u_inlet);
                v_jp10 = get_field_value(y, 1, i, j + 1, k, xN, yN, zN, u_inlet);
                v_jm10 = get_field_value(y, 1, i, j - 1, k, xN, yN, zN, u_inlet);
                v_kp1 = get_field_value(y, 1, i, j, k + 1, xN, yN, zN, u_inlet);
                v_km1 = get_field_value(y, 1, i, j, k - 1, xN, yN, zN, u_inlet);
                v_ijk = get_field_value(y, 1, i, j, k, xN, yN, zN, u_inlet);
                u_ip10 = get_field_value(y, 0, i + 1, j, k, xN, yN, zN, u_inlet);
                u_im10 = get_field_value(y, 0, i - 1, j, k, xN, yN, zN, u_inlet);
                w_kp10 = get_field_value(y, 2, i, j, k + 1, xN, yN, zN, u_inlet);
                w_km10 = get_field_value(y, 2, i, j, k - 1, xN, yN, zN, u_inlet);
                p_jp1 = get_field_value(y, 3, i, j + 1, k, xN, yN, zN, u_inlet);
                p_ijk0 = get_field_value(y, 3, i, j, k, xN, yN, zN, u_inlet);
                conv_x0 = 0.5 * dy * dz * (u_ip10 * v_ip1 - u_im10 * v_im1);
                conv_y0 = 0.5 * dx * dz * (v_jp10 * v_jp10 - v_jm10 * v_jm10);
                conv_z0 = 0.5 * dx * dy * (v_kp1 * w_kp10 - v_km1 * w_km10);
                pres0 = (dx * dz) * (p_jp1 - p_ijk0);
                diff0 = (1. / Re) * ((dy * dz / dx) * (v_ip1 - 2. * v_ijk + v_im1) + (dx * dz / dy) * (v_jp10 - 2. * v_ijk + v_jm10) + (dx * dy / dz) * (v_kp1 - 2. * v_ijk + v_km1));
                result = conv_x0 + conv_y0 + conv_z0 + pres0 - diff0;
            } else {
                _cond2 = eq_type == 2;
                if (_cond2) {
                    w_ip1 = get_field_value(y, 2, i + 1, j, k, xN, yN, zN, u_inlet);
                    w_im1 = get_field_value(y, 2, i - 1, j, k, xN, yN, zN, u_inlet);
                    w_jp1 = get_field_value(y, 2, i, j + 1, k, xN, yN, zN, u_inlet);
                    w_jm1 = get_field_value(y, 2, i, j - 1, k, xN, yN, zN, u_inlet);
                    w_kp11 = get_field_value(y, 2, i, j, k + 1, xN, yN, zN, u_inlet);
                    w_km11 = get_field_value(y, 2, i, j, k - 1, xN, yN, zN, u_inlet);
                    w_ijk = get_field_value(y, 2, i, j, k, xN, yN, zN, u_inlet);
                    u_ip11 = get_field_value(y, 0, i + 1, j, k, xN, yN, zN, u_inlet);
                    u_im11 = get_field_value(y, 0, i - 1, j, k, xN, yN, zN, u_inlet);
                    v_jp11 = get_field_value(y, 1, i, j + 1, k, xN, yN, zN, u_inlet);
                    v_jm11 = get_field_value(y, 1, i, j - 1, k, xN, yN, zN, u_inlet);
                    p_kp1 = get_field_value(y, 3, i, j, k + 1, xN, yN, zN, u_inlet);
                    p_ijk1 = get_field_value(y, 3, i, j, k, xN, yN, zN, u_inlet);
                    conv_x1 = 0.5 * dy * dz * (u_ip11 * w_ip1 - u_im11 * w_im1);
                    conv_y1 = 0.5 * dx * dz * (v_jp11 * w_jp1 - v_jm11 * w_jm1);
                    conv_z1 = 0.5 * dx * dy * (w_kp11 * w_kp11 - w_km11 * w_km11);
                    pres1 = (dx * dy) * (p_kp1 - p_ijk1);
                    diff1 = (1. / Re) * ((dy * dz / dx) * (w_ip1 - 2. * w_ijk + w_im1) + (dx * dz / dy) * (w_jp1 - 2. * w_ijk + w_jm1) + (dx * dy / dz) * (w_kp11 - 2. * w_ijk + w_km11));
                    result = conv_x1 + conv_y1 + conv_z1 + pres1 - diff1;
                } else {
                    _cond3 = eq_type == 3;
                    if (_cond3) {
                        u_ip12 = get_field_value(y, 0, i + 1, j, k, xN, yN, zN, u_inlet);
                        u_im12 = get_field_value(y, 0, i - 1, j, k, xN, yN, zN, u_inlet);
                        v_jp12 = get_field_value(y, 1, i, j + 1, k, xN, yN, zN, u_inlet);
                        v_jm12 = get_field_value(y, 1, i, j - 1, k, xN, yN, zN, u_inlet);
                        w_kp12 = get_field_value(y, 2, i, j, k + 1, xN, yN, zN, u_inlet);
                        w_km12 = get_field_value(y, 2, i, j, k - 1, xN, yN, zN, u_inlet);
                        cont = (dy * dz / 2.) * (u_ip12 - u_im12) + (dx * dz / 2.) * (v_jp12 - v_jm12) + (dx * dy / 2.) * (w_kp12 - w_km12);
                        result = cont;
                    }
                }
            }
        }
    }
    _d_result += 1;
    if (_cond0) {
        {
            _d_conv_x += _d_result;
            _d_conv_y += _d_result;
            _d_conv_z += _d_result;
            _d_pres += _d_result;
            _d_diff += -_d_result;
            _d_result = 0.F;
        }
        {
            double _r91 = _d_diff * ((dy * dz / dx) * (u_ip1 - 2. * u_ijk + u_im1) + (dx * dz / dy) * (u_jp1 - 2. * u_ijk + u_jm1) + (dx * dy / dz) * (u_kp1 - 2. * u_ijk + u_km1)) * -(1. / (Re * Re));
            _d_Re += _r91;
            _d_dy += (1. / Re) * _d_diff * (u_ip1 - 2. * u_ijk + u_im1) / dx * dz;
            _d_dz += dy * (1. / Re) * _d_diff * (u_ip1 - 2. * u_ijk + u_im1) / dx;
            double _r92 = (1. / Re) * _d_diff * (u_ip1 - 2. * u_ijk + u_im1) * -(dy * dz / (dx * dx));
            _d_dx += _r92;
            _d_u_ip1 += (dy * dz / dx) * (1. / Re) * _d_diff;
            _d_u_ijk += 2. * -(dy * dz / dx) * (1. / Re) * _d_diff;
            _d_u_im1 += (dy * dz / dx) * (1. / Re) * _d_diff;
            _d_dx += (1. / Re) * _d_diff * (u_jp1 - 2. * u_ijk + u_jm1) / dy * dz;
            _d_dz += dx * (1. / Re) * _d_diff * (u_jp1 - 2. * u_ijk + u_jm1) / dy;
            double _r93 = (1. / Re) * _d_diff * (u_jp1 - 2. * u_ijk + u_jm1) * -(dx * dz / (dy * dy));
            _d_dy += _r93;
            _d_u_jp1 += (dx * dz / dy) * (1. / Re) * _d_diff;
            _d_u_ijk += 2. * -(dx * dz / dy) * (1. / Re) * _d_diff;
            _d_u_jm1 += (dx * dz / dy) * (1. / Re) * _d_diff;
            _d_dx += (1. / Re) * _d_diff * (u_kp1 - 2. * u_ijk + u_km1) / dz * dy;
            _d_dy += dx * (1. / Re) * _d_diff * (u_kp1 - 2. * u_ijk + u_km1) / dz;
            double _r94 = (1. / Re) * _d_diff * (u_kp1 - 2. * u_ijk + u_km1) * -(dx * dy / (dz * dz));
            _d_dz += _r94;
            _d_u_kp1 += (dx * dy / dz) * (1. / Re) * _d_diff;
            _d_u_ijk += 2. * -(dx * dy / dz) * (1. / Re) * _d_diff;
            _d_u_km1 += (dx * dy / dz) * (1. / Re) * _d_diff;
        }
        {
            _d_dy += _d_pres * (p_ip1 - p_ijk) * dz;
            _d_dz += dy * _d_pres * (p_ip1 - p_ijk);
            _d_p_ip1 += (dy * dz) * _d_pres;
            _d_p_ijk += -(dy * dz) * _d_pres;
        }
        {
            _d_dx += 0.5 * _d_conv_z * (u_kp1 * w_kp1 - u_km1 * w_km1) * dy;
            _d_dy += 0.5 * dx * _d_conv_z * (u_kp1 * w_kp1 - u_km1 * w_km1);
            _d_u_kp1 += 0.5 * dx * dy * _d_conv_z * w_kp1;
            _d_w_kp1 += u_kp1 * 0.5 * dx * dy * _d_conv_z;
            _d_u_km1 += -0.5 * dx * dy * _d_conv_z * w_km1;
            _d_w_km1 += u_km1 * -0.5 * dx * dy * _d_conv_z;
        }
        {
            _d_dx += 0.5 * _d_conv_y * (u_jp1 * v_jp1 - u_jm1 * v_jm1) * dz;
            _d_dz += 0.5 * dx * _d_conv_y * (u_jp1 * v_jp1 - u_jm1 * v_jm1);
            _d_u_jp1 += 0.5 * dx * dz * _d_conv_y * v_jp1;
            _d_v_jp1 += u_jp1 * 0.5 * dx * dz * _d_conv_y;
            _d_u_jm1 += -0.5 * dx * dz * _d_conv_y * v_jm1;
            _d_v_jm1 += u_jm1 * -0.5 * dx * dz * _d_conv_y;
        }
        {
            _d_dy += 0.5 * _d_conv_x * (u_ip1 * u_ip1 - u_im1 * u_im1) * dz;
            _d_dz += 0.5 * dy * _d_conv_x * (u_ip1 * u_ip1 - u_im1 * u_im1);
            _d_u_ip1 += 0.5 * dy * dz * _d_conv_x * u_ip1;
            _d_u_ip1 += u_ip1 * 0.5 * dy * dz * _d_conv_x;
            _d_u_im1 += -0.5 * dy * dz * _d_conv_x * u_im1;
            _d_u_im1 += u_im1 * -0.5 * dy * dz * _d_conv_x;
        }
        {
            int _r84 = 0;
            int _r85 = 0;
            int _r86 = 0;
            int _r87 = 0;
            int _r88 = 0;
            int _r89 = 0;
            int _r90 = 0;
            get_field_value_pullback(y, 3, i, j, k, xN, yN, zN, u_inlet, _d_p_ijk, _d_y, &_r84, &_r85, &_r86, &_r87, &_r88, &_r89, &_r90);
            _d_i += _r85;
            _d_j += _r86;
            _d_k += _r87;
            _d_xN += _r88;
            _d_yN += _r89;
            _d_zN += _r90;
        }
        {
            int _r77 = 0;
            int _r78 = 0;
            int _r79 = 0;
            int _r80 = 0;
            int _r81 = 0;
            int _r82 = 0;
            int _r83 = 0;
            get_field_value_pullback(y, 3, i + 1, j, k, xN, yN, zN, u_inlet, _d_p_ip1, _d_y, &_r77, &_r78, &_r79, &_r80, &_r81, &_r82, &_r83);
            _d_i += _r78;
            _d_j += _r79;
            _d_k += _r80;
            _d_xN += _r81;
            _d_yN += _r82;
            _d_zN += _r83;
        }
        {
            int _r70 = 0;
            int _r71 = 0;
            int _r72 = 0;
            int _r73 = 0;
            int _r74 = 0;
            int _r75 = 0;
            int _r76 = 0;
            get_field_value_pullback(y, 2, i, j, k - 1, xN, yN, zN, u_inlet, _d_w_km1, _d_y, &_r70, &_r71, &_r72, &_r73, &_r74, &_r75, &_r76);
            _d_i += _r71;
            _d_j += _r72;
            _d_k += _r73;
            _d_xN += _r74;
            _d_yN += _r75;
            _d_zN += _r76;
        }
        {
            int _r63 = 0;
            int _r64 = 0;
            int _r65 = 0;
            int _r66 = 0;
            int _r67 = 0;
            int _r68 = 0;
            int _r69 = 0;
            get_field_value_pullback(y, 2, i, j, k + 1, xN, yN, zN, u_inlet, _d_w_kp1, _d_y, &_r63, &_r64, &_r65, &_r66, &_r67, &_r68, &_r69);
            _d_i += _r64;
            _d_j += _r65;
            _d_k += _r66;
            _d_xN += _r67;
            _d_yN += _r68;
            _d_zN += _r69;
        }
        {
            int _r56 = 0;
            int _r57 = 0;
            int _r58 = 0;
            int _r59 = 0;
            int _r60 = 0;
            int _r61 = 0;
            int _r62 = 0;
            get_field_value_pullback(y, 1, i, j - 1, k, xN, yN, zN, u_inlet, _d_v_jm1, _d_y, &_r56, &_r57, &_r58, &_r59, &_r60, &_r61, &_r62);
            _d_i += _r57;
            _d_j += _r58;
            _d_k += _r59;
            _d_xN += _r60;
            _d_yN += _r61;
            _d_zN += _r62;
        }
        {
            int _r49 = 0;
            int _r50 = 0;
            int _r51 = 0;
            int _r52 = 0;
            int _r53 = 0;
            int _r54 = 0;
            int _r55 = 0;
            get_field_value_pullback(y, 1, i, j + 1, k, xN, yN, zN, u_inlet, _d_v_jp1, _d_y, &_r49, &_r50, &_r51, &_r52, &_r53, &_r54, &_r55);
            _d_i += _r50;
            _d_j += _r51;
            _d_k += _r52;
            _d_xN += _r53;
            _d_yN += _r54;
            _d_zN += _r55;
        }
        {
            int _r42 = 0;
            int _r43 = 0;
            int _r44 = 0;
            int _r45 = 0;
            int _r46 = 0;
            int _r47 = 0;
            int _r48 = 0;
            get_field_value_pullback(y, 0, i, j, k, xN, yN, zN, u_inlet, _d_u_ijk, _d_y, &_r42, &_r43, &_r44, &_r45, &_r46, &_r47, &_r48);
            _d_i += _r43;
            _d_j += _r44;
            _d_k += _r45;
            _d_xN += _r46;
            _d_yN += _r47;
            _d_zN += _r48;
        }
        {
            int _r35 = 0;
            int _r36 = 0;
            int _r37 = 0;
            int _r38 = 0;
            int _r39 = 0;
            int _r40 = 0;
            int _r41 = 0;
            get_field_value_pullback(y, 0, i, j, k - 1, xN, yN, zN, u_inlet, _d_u_km1, _d_y, &_r35, &_r36, &_r37, &_r38, &_r39, &_r40, &_r41);
            _d_i += _r36;
            _d_j += _r37;
            _d_k += _r38;
            _d_xN += _r39;
            _d_yN += _r40;
            _d_zN += _r41;
        }
        {
            int _r28 = 0;
            int _r29 = 0;
            int _r30 = 0;
            int _r31 = 0;
            int _r32 = 0;
            int _r33 = 0;
            int _r34 = 0;
            get_field_value_pullback(y, 0, i, j, k + 1, xN, yN, zN, u_inlet, _d_u_kp1, _d_y, &_r28, &_r29, &_r30, &_r31, &_r32, &_r33, &_r34);
            _d_i += _r29;
            _d_j += _r30;
            _d_k += _r31;
            _d_xN += _r32;
            _d_yN += _r33;
            _d_zN += _r34;
        }
        {
            int _r21 = 0;
            int _r22 = 0;
            int _r23 = 0;
            int _r24 = 0;
            int _r25 = 0;
            int _r26 = 0;
            int _r27 = 0;
            get_field_value_pullback(y, 0, i, j - 1, k, xN, yN, zN, u_inlet, _d_u_jm1, _d_y, &_r21, &_r22, &_r23, &_r24, &_r25, &_r26, &_r27);
            _d_i += _r22;
            _d_j += _r23;
            _d_k += _r24;
            _d_xN += _r25;
            _d_yN += _r26;
            _d_zN += _r27;
        }
        {
            int _r14 = 0;
            int _r15 = 0;
            int _r16 = 0;
            int _r17 = 0;
            int _r18 = 0;
            int _r19 = 0;
            int _r20 = 0;
            get_field_value_pullback(y, 0, i, j + 1, k, xN, yN, zN, u_inlet, _d_u_jp1, _d_y, &_r14, &_r15, &_r16, &_r17, &_r18, &_r19, &_r20);
            _d_i += _r15;
            _d_j += _r16;
            _d_k += _r17;
            _d_xN += _r18;
            _d_yN += _r19;
            _d_zN += _r20;
        }
        {
            int _r7 = 0;
            int _r8 = 0;
            int _r9 = 0;
            int _r10 = 0;
            int _r11 = 0;
            int _r12 = 0;
            int _r13 = 0;
            get_field_value_pullback(y, 0, i - 1, j, k, xN, yN, zN, u_inlet, _d_u_im1, _d_y, &_r7, &_r8, &_r9, &_r10, &_r11, &_r12, &_r13);
            _d_i += _r8;
            _d_j += _r9;
            _d_k += _r10;
            _d_xN += _r11;
            _d_yN += _r12;
            _d_zN += _r13;
        }
        {
            int _r0 = 0;
            int _r1 = 0;
            int _r2 = 0;
            int _r3 = 0;
            int _r4 = 0;
            int _r5 = 0;
            int _r6 = 0;
            get_field_value_pullback(y, 0, i + 1, j, k, xN, yN, zN, u_inlet, _d_u_ip1, _d_y, &_r0, &_r1, &_r2, &_r3, &_r4, &_r5, &_r6);
            _d_i += _r1;
            _d_j += _r2;
            _d_k += _r3;
            _d_xN += _r4;
            _d_yN += _r5;
            _d_zN += _r6;
        }
    } else if (_cond1) {
        {
            _d_conv_x0 += _d_result;
            _d_conv_y0 += _d_result;
            _d_conv_z0 += _d_result;
            _d_pres0 += _d_result;
            _d_diff0 += -_d_result;
            _d_result = 0.F;
        }
        {
            double _r186 = _d_diff0 * ((dy * dz / dx) * (v_ip1 - 2. * v_ijk + v_im1) + (dx * dz / dy) * (v_jp10 - 2. * v_ijk + v_jm10) + (dx * dy / dz) * (v_kp1 - 2. * v_ijk + v_km1)) * -(1. / (Re * Re));
            _d_Re += _r186;
            _d_dy += (1. / Re) * _d_diff0 * (v_ip1 - 2. * v_ijk + v_im1) / dx * dz;
            _d_dz += dy * (1. / Re) * _d_diff0 * (v_ip1 - 2. * v_ijk + v_im1) / dx;
            double _r187 = (1. / Re) * _d_diff0 * (v_ip1 - 2. * v_ijk + v_im1) * -(dy * dz / (dx * dx));
            _d_dx += _r187;
            _d_v_ip1 += (dy * dz / dx) * (1. / Re) * _d_diff0;
            _d_v_ijk += 2. * -(dy * dz / dx) * (1. / Re) * _d_diff0;
            _d_v_im1 += (dy * dz / dx) * (1. / Re) * _d_diff0;
            _d_dx += (1. / Re) * _d_diff0 * (v_jp10 - 2. * v_ijk + v_jm10) / dy * dz;
            _d_dz += dx * (1. / Re) * _d_diff0 * (v_jp10 - 2. * v_ijk + v_jm10) / dy;
            double _r188 = (1. / Re) * _d_diff0 * (v_jp10 - 2. * v_ijk + v_jm10) * -(dx * dz / (dy * dy));
            _d_dy += _r188;
            _d_v_jp10 += (dx * dz / dy) * (1. / Re) * _d_diff0;
            _d_v_ijk += 2. * -(dx * dz / dy) * (1. / Re) * _d_diff0;
            _d_v_jm10 += (dx * dz / dy) * (1. / Re) * _d_diff0;
            _d_dx += (1. / Re) * _d_diff0 * (v_kp1 - 2. * v_ijk + v_km1) / dz * dy;
            _d_dy += dx * (1. / Re) * _d_diff0 * (v_kp1 - 2. * v_ijk + v_km1) / dz;
            double _r189 = (1. / Re) * _d_diff0 * (v_kp1 - 2. * v_ijk + v_km1) * -(dx * dy / (dz * dz));
            _d_dz += _r189;
            _d_v_kp1 += (dx * dy / dz) * (1. / Re) * _d_diff0;
            _d_v_ijk += 2. * -(dx * dy / dz) * (1. / Re) * _d_diff0;
            _d_v_km1 += (dx * dy / dz) * (1. / Re) * _d_diff0;
        }
        {
            _d_dx += _d_pres0 * (p_jp1 - p_ijk0) * dz;
            _d_dz += dx * _d_pres0 * (p_jp1 - p_ijk0);
            _d_p_jp1 += (dx * dz) * _d_pres0;
            _d_p_ijk0 += -(dx * dz) * _d_pres0;
        }
        {
            _d_dx += 0.5 * _d_conv_z0 * (v_kp1 * w_kp10 - v_km1 * w_km10) * dy;
            _d_dy += 0.5 * dx * _d_conv_z0 * (v_kp1 * w_kp10 - v_km1 * w_km10);
            _d_v_kp1 += 0.5 * dx * dy * _d_conv_z0 * w_kp10;
            _d_w_kp10 += v_kp1 * 0.5 * dx * dy * _d_conv_z0;
            _d_v_km1 += -0.5 * dx * dy * _d_conv_z0 * w_km10;
            _d_w_km10 += v_km1 * -0.5 * dx * dy * _d_conv_z0;
        }
        {
            _d_dx += 0.5 * _d_conv_y0 * (v_jp10 * v_jp10 - v_jm10 * v_jm10) * dz;
            _d_dz += 0.5 * dx * _d_conv_y0 * (v_jp10 * v_jp10 - v_jm10 * v_jm10);
            _d_v_jp10 += 0.5 * dx * dz * _d_conv_y0 * v_jp10;
            _d_v_jp10 += v_jp10 * 0.5 * dx * dz * _d_conv_y0;
            _d_v_jm10 += -0.5 * dx * dz * _d_conv_y0 * v_jm10;
            _d_v_jm10 += v_jm10 * -0.5 * dx * dz * _d_conv_y0;
        }
        {
            _d_dy += 0.5 * _d_conv_x0 * (u_ip10 * v_ip1 - u_im10 * v_im1) * dz;
            _d_dz += 0.5 * dy * _d_conv_x0 * (u_ip10 * v_ip1 - u_im10 * v_im1);
            _d_u_ip10 += 0.5 * dy * dz * _d_conv_x0 * v_ip1;
            _d_v_ip1 += u_ip10 * 0.5 * dy * dz * _d_conv_x0;
            _d_u_im10 += -0.5 * dy * dz * _d_conv_x0 * v_im1;
            _d_v_im1 += u_im10 * -0.5 * dy * dz * _d_conv_x0;
        }
        {
            int _r179 = 0;
            int _r180 = 0;
            int _r181 = 0;
            int _r182 = 0;
            int _r183 = 0;
            int _r184 = 0;
            int _r185 = 0;
            get_field_value_pullback(y, 3, i, j, k, xN, yN, zN, u_inlet, _d_p_ijk0, _d_y, &_r179, &_r180, &_r181, &_r182, &_r183, &_r184, &_r185);
            _d_i += _r180;
            _d_j += _r181;
            _d_k += _r182;
            _d_xN += _r183;
            _d_yN += _r184;
            _d_zN += _r185;
        }
        {
            int _r172 = 0;
            int _r173 = 0;
            int _r174 = 0;
            int _r175 = 0;
            int _r176 = 0;
            int _r177 = 0;
            int _r178 = 0;
            get_field_value_pullback(y, 3, i, j + 1, k, xN, yN, zN, u_inlet, _d_p_jp1, _d_y, &_r172, &_r173, &_r174, &_r175, &_r176, &_r177, &_r178);
            _d_i += _r173;
            _d_j += _r174;
            _d_k += _r175;
            _d_xN += _r176;
            _d_yN += _r177;
            _d_zN += _r178;
        }
        {
            int _r165 = 0;
            int _r166 = 0;
            int _r167 = 0;
            int _r168 = 0;
            int _r169 = 0;
            int _r170 = 0;
            int _r171 = 0;
            get_field_value_pullback(y, 2, i, j, k - 1, xN, yN, zN, u_inlet, _d_w_km10, _d_y, &_r165, &_r166, &_r167, &_r168, &_r169, &_r170, &_r171);
            _d_i += _r166;
            _d_j += _r167;
            _d_k += _r168;
            _d_xN += _r169;
            _d_yN += _r170;
            _d_zN += _r171;
        }
        {
            int _r158 = 0;
            int _r159 = 0;
            int _r160 = 0;
            int _r161 = 0;
            int _r162 = 0;
            int _r163 = 0;
            int _r164 = 0;
            get_field_value_pullback(y, 2, i, j, k + 1, xN, yN, zN, u_inlet, _d_w_kp10, _d_y, &_r158, &_r159, &_r160, &_r161, &_r162, &_r163, &_r164);
            _d_i += _r159;
            _d_j += _r160;
            _d_k += _r161;
            _d_xN += _r162;
            _d_yN += _r163;
            _d_zN += _r164;
        }
        {
            int _r151 = 0;
            int _r152 = 0;
            int _r153 = 0;
            int _r154 = 0;
            int _r155 = 0;
            int _r156 = 0;
            int _r157 = 0;
            get_field_value_pullback(y, 0, i - 1, j, k, xN, yN, zN, u_inlet, _d_u_im10, _d_y, &_r151, &_r152, &_r153, &_r154, &_r155, &_r156, &_r157);
            _d_i += _r152;
            _d_j += _r153;
            _d_k += _r154;
            _d_xN += _r155;
            _d_yN += _r156;
            _d_zN += _r157;
        }
        {
            int _r144 = 0;
            int _r145 = 0;
            int _r146 = 0;
            int _r147 = 0;
            int _r148 = 0;
            int _r149 = 0;
            int _r150 = 0;
            get_field_value_pullback(y, 0, i + 1, j, k, xN, yN, zN, u_inlet, _d_u_ip10, _d_y, &_r144, &_r145, &_r146, &_r147, &_r148, &_r149, &_r150);
            _d_i += _r145;
            _d_j += _r146;
            _d_k += _r147;
            _d_xN += _r148;
            _d_yN += _r149;
            _d_zN += _r150;
        }
        {
            int _r137 = 0;
            int _r138 = 0;
            int _r139 = 0;
            int _r140 = 0;
            int _r141 = 0;
            int _r142 = 0;
            int _r143 = 0;
            get_field_value_pullback(y, 1, i, j, k, xN, yN, zN, u_inlet, _d_v_ijk, _d_y, &_r137, &_r138, &_r139, &_r140, &_r141, &_r142, &_r143);
            _d_i += _r138;
            _d_j += _r139;
            _d_k += _r140;
            _d_xN += _r141;
            _d_yN += _r142;
            _d_zN += _r143;
        }
        {
            int _r130 = 0;
            int _r131 = 0;
            int _r132 = 0;
            int _r133 = 0;
            int _r134 = 0;
            int _r135 = 0;
            int _r136 = 0;
            get_field_value_pullback(y, 1, i, j, k - 1, xN, yN, zN, u_inlet, _d_v_km1, _d_y, &_r130, &_r131, &_r132, &_r133, &_r134, &_r135, &_r136);
            _d_i += _r131;
            _d_j += _r132;
            _d_k += _r133;
            _d_xN += _r134;
            _d_yN += _r135;
            _d_zN += _r136;
        }
        {
            int _r123 = 0;
            int _r124 = 0;
            int _r125 = 0;
            int _r126 = 0;
            int _r127 = 0;
            int _r128 = 0;
            int _r129 = 0;
            get_field_value_pullback(y, 1, i, j, k + 1, xN, yN, zN, u_inlet, _d_v_kp1, _d_y, &_r123, &_r124, &_r125, &_r126, &_r127, &_r128, &_r129);
            _d_i += _r124;
            _d_j += _r125;
            _d_k += _r126;
            _d_xN += _r127;
            _d_yN += _r128;
            _d_zN += _r129;
        }
        {
            int _r116 = 0;
            int _r117 = 0;
            int _r118 = 0;
            int _r119 = 0;
            int _r120 = 0;
            int _r121 = 0;
            int _r122 = 0;
            get_field_value_pullback(y, 1, i, j - 1, k, xN, yN, zN, u_inlet, _d_v_jm10, _d_y, &_r116, &_r117, &_r118, &_r119, &_r120, &_r121, &_r122);
            _d_i += _r117;
            _d_j += _r118;
            _d_k += _r119;
            _d_xN += _r120;
            _d_yN += _r121;
            _d_zN += _r122;
        }
        {
            int _r109 = 0;
            int _r110 = 0;
            int _r111 = 0;
            int _r112 = 0;
            int _r113 = 0;
            int _r114 = 0;
            int _r115 = 0;
            get_field_value_pullback(y, 1, i, j + 1, k, xN, yN, zN, u_inlet, _d_v_jp10, _d_y, &_r109, &_r110, &_r111, &_r112, &_r113, &_r114, &_r115);
            _d_i += _r110;
            _d_j += _r111;
            _d_k += _r112;
            _d_xN += _r113;
            _d_yN += _r114;
            _d_zN += _r115;
        }
        {
            int _r102 = 0;
            int _r103 = 0;
            int _r104 = 0;
            int _r105 = 0;
            int _r106 = 0;
            int _r107 = 0;
            int _r108 = 0;
            get_field_value_pullback(y, 1, i - 1, j, k, xN, yN, zN, u_inlet, _d_v_im1, _d_y, &_r102, &_r103, &_r104, &_r105, &_r106, &_r107, &_r108);
            _d_i += _r103;
            _d_j += _r104;
            _d_k += _r105;
            _d_xN += _r106;
            _d_yN += _r107;
            _d_zN += _r108;
        }
        {
            int _r95 = 0;
            int _r96 = 0;
            int _r97 = 0;
            int _r98 = 0;
            int _r99 = 0;
            int _r100 = 0;
            int _r101 = 0;
            get_field_value_pullback(y, 1, i + 1, j, k, xN, yN, zN, u_inlet, _d_v_ip1, _d_y, &_r95, &_r96, &_r97, &_r98, &_r99, &_r100, &_r101);
            _d_i += _r96;
            _d_j += _r97;
            _d_k += _r98;
            _d_xN += _r99;
            _d_yN += _r100;
            _d_zN += _r101;
        }
    } else if (_cond2) {
        {
            _d_conv_x1 += _d_result;
            _d_conv_y1 += _d_result;
            _d_conv_z1 += _d_result;
            _d_pres1 += _d_result;
            _d_diff1 += -_d_result;
            _d_result = 0.F;
        }
        {
            double _r281 = _d_diff1 * ((dy * dz / dx) * (w_ip1 - 2. * w_ijk + w_im1) + (dx * dz / dy) * (w_jp1 - 2. * w_ijk + w_jm1) + (dx * dy / dz) * (w_kp11 - 2. * w_ijk + w_km11)) * -(1. / (Re * Re));
            _d_Re += _r281;
            _d_dy += (1. / Re) * _d_diff1 * (w_ip1 - 2. * w_ijk + w_im1) / dx * dz;
            _d_dz += dy * (1. / Re) * _d_diff1 * (w_ip1 - 2. * w_ijk + w_im1) / dx;
            double _r282 = (1. / Re) * _d_diff1 * (w_ip1 - 2. * w_ijk + w_im1) * -(dy * dz / (dx * dx));
            _d_dx += _r282;
            _d_w_ip1 += (dy * dz / dx) * (1. / Re) * _d_diff1;
            _d_w_ijk += 2. * -(dy * dz / dx) * (1. / Re) * _d_diff1;
            _d_w_im1 += (dy * dz / dx) * (1. / Re) * _d_diff1;
            _d_dx += (1. / Re) * _d_diff1 * (w_jp1 - 2. * w_ijk + w_jm1) / dy * dz;
            _d_dz += dx * (1. / Re) * _d_diff1 * (w_jp1 - 2. * w_ijk + w_jm1) / dy;
            double _r283 = (1. / Re) * _d_diff1 * (w_jp1 - 2. * w_ijk + w_jm1) * -(dx * dz / (dy * dy));
            _d_dy += _r283;
            _d_w_jp1 += (dx * dz / dy) * (1. / Re) * _d_diff1;
            _d_w_ijk += 2. * -(dx * dz / dy) * (1. / Re) * _d_diff1;
            _d_w_jm1 += (dx * dz / dy) * (1. / Re) * _d_diff1;
            _d_dx += (1. / Re) * _d_diff1 * (w_kp11 - 2. * w_ijk + w_km11) / dz * dy;
            _d_dy += dx * (1. / Re) * _d_diff1 * (w_kp11 - 2. * w_ijk + w_km11) / dz;
            double _r284 = (1. / Re) * _d_diff1 * (w_kp11 - 2. * w_ijk + w_km11) * -(dx * dy / (dz * dz));
            _d_dz += _r284;
            _d_w_kp11 += (dx * dy / dz) * (1. / Re) * _d_diff1;
            _d_w_ijk += 2. * -(dx * dy / dz) * (1. / Re) * _d_diff1;
            _d_w_km11 += (dx * dy / dz) * (1. / Re) * _d_diff1;
        }
        {
            _d_dx += _d_pres1 * (p_kp1 - p_ijk1) * dy;
            _d_dy += dx * _d_pres1 * (p_kp1 - p_ijk1);
            _d_p_kp1 += (dx * dy) * _d_pres1;
            _d_p_ijk1 += -(dx * dy) * _d_pres1;
        }
        {
            _d_dx += 0.5 * _d_conv_z1 * (w_kp11 * w_kp11 - w_km11 * w_km11) * dy;
            _d_dy += 0.5 * dx * _d_conv_z1 * (w_kp11 * w_kp11 - w_km11 * w_km11);
            _d_w_kp11 += 0.5 * dx * dy * _d_conv_z1 * w_kp11;
            _d_w_kp11 += w_kp11 * 0.5 * dx * dy * _d_conv_z1;
            _d_w_km11 += -0.5 * dx * dy * _d_conv_z1 * w_km11;
            _d_w_km11 += w_km11 * -0.5 * dx * dy * _d_conv_z1;
        }
        {
            _d_dx += 0.5 * _d_conv_y1 * (v_jp11 * w_jp1 - v_jm11 * w_jm1) * dz;
            _d_dz += 0.5 * dx * _d_conv_y1 * (v_jp11 * w_jp1 - v_jm11 * w_jm1);
            _d_v_jp11 += 0.5 * dx * dz * _d_conv_y1 * w_jp1;
            _d_w_jp1 += v_jp11 * 0.5 * dx * dz * _d_conv_y1;
            _d_v_jm11 += -0.5 * dx * dz * _d_conv_y1 * w_jm1;
            _d_w_jm1 += v_jm11 * -0.5 * dx * dz * _d_conv_y1;
        }
        {
            _d_dy += 0.5 * _d_conv_x1 * (u_ip11 * w_ip1 - u_im11 * w_im1) * dz;
            _d_dz += 0.5 * dy * _d_conv_x1 * (u_ip11 * w_ip1 - u_im11 * w_im1);
            _d_u_ip11 += 0.5 * dy * dz * _d_conv_x1 * w_ip1;
            _d_w_ip1 += u_ip11 * 0.5 * dy * dz * _d_conv_x1;
            _d_u_im11 += -0.5 * dy * dz * _d_conv_x1 * w_im1;
            _d_w_im1 += u_im11 * -0.5 * dy * dz * _d_conv_x1;
        }
        {
            int _r274 = 0;
            int _r275 = 0;
            int _r276 = 0;
            int _r277 = 0;
            int _r278 = 0;
            int _r279 = 0;
            int _r280 = 0;
            get_field_value_pullback(y, 3, i, j, k, xN, yN, zN, u_inlet, _d_p_ijk1, _d_y, &_r274, &_r275, &_r276, &_r277, &_r278, &_r279, &_r280);
            _d_i += _r275;
            _d_j += _r276;
            _d_k += _r277;
            _d_xN += _r278;
            _d_yN += _r279;
            _d_zN += _r280;
        }
        {
            int _r267 = 0;
            int _r268 = 0;
            int _r269 = 0;
            int _r270 = 0;
            int _r271 = 0;
            int _r272 = 0;
            int _r273 = 0;
            get_field_value_pullback(y, 3, i, j, k + 1, xN, yN, zN, u_inlet, _d_p_kp1, _d_y, &_r267, &_r268, &_r269, &_r270, &_r271, &_r272, &_r273);
            _d_i += _r268;
            _d_j += _r269;
            _d_k += _r270;
            _d_xN += _r271;
            _d_yN += _r272;
            _d_zN += _r273;
        }
        {
            int _r260 = 0;
            int _r261 = 0;
            int _r262 = 0;
            int _r263 = 0;
            int _r264 = 0;
            int _r265 = 0;
            int _r266 = 0;
            get_field_value_pullback(y, 1, i, j - 1, k, xN, yN, zN, u_inlet, _d_v_jm11, _d_y, &_r260, &_r261, &_r262, &_r263, &_r264, &_r265, &_r266);
            _d_i += _r261;
            _d_j += _r262;
            _d_k += _r263;
            _d_xN += _r264;
            _d_yN += _r265;
            _d_zN += _r266;
        }
        {
            int _r253 = 0;
            int _r254 = 0;
            int _r255 = 0;
            int _r256 = 0;
            int _r257 = 0;
            int _r258 = 0;
            int _r259 = 0;
            get_field_value_pullback(y, 1, i, j + 1, k, xN, yN, zN, u_inlet, _d_v_jp11, _d_y, &_r253, &_r254, &_r255, &_r256, &_r257, &_r258, &_r259);
            _d_i += _r254;
            _d_j += _r255;
            _d_k += _r256;
            _d_xN += _r257;
            _d_yN += _r258;
            _d_zN += _r259;
        }
        {
            int _r246 = 0;
            int _r247 = 0;
            int _r248 = 0;
            int _r249 = 0;
            int _r250 = 0;
            int _r251 = 0;
            int _r252 = 0;
            get_field_value_pullback(y, 0, i - 1, j, k, xN, yN, zN, u_inlet, _d_u_im11, _d_y, &_r246, &_r247, &_r248, &_r249, &_r250, &_r251, &_r252);
            _d_i += _r247;
            _d_j += _r248;
            _d_k += _r249;
            _d_xN += _r250;
            _d_yN += _r251;
            _d_zN += _r252;
        }
        {
            int _r239 = 0;
            int _r240 = 0;
            int _r241 = 0;
            int _r242 = 0;
            int _r243 = 0;
            int _r244 = 0;
            int _r245 = 0;
            get_field_value_pullback(y, 0, i + 1, j, k, xN, yN, zN, u_inlet, _d_u_ip11, _d_y, &_r239, &_r240, &_r241, &_r242, &_r243, &_r244, &_r245);
            _d_i += _r240;
            _d_j += _r241;
            _d_k += _r242;
            _d_xN += _r243;
            _d_yN += _r244;
            _d_zN += _r245;
        }
        {
            int _r232 = 0;
            int _r233 = 0;
            int _r234 = 0;
            int _r235 = 0;
            int _r236 = 0;
            int _r237 = 0;
            int _r238 = 0;
            get_field_value_pullback(y, 2, i, j, k, xN, yN, zN, u_inlet, _d_w_ijk, _d_y, &_r232, &_r233, &_r234, &_r235, &_r236, &_r237, &_r238);
            _d_i += _r233;
            _d_j += _r234;
            _d_k += _r235;
            _d_xN += _r236;
            _d_yN += _r237;
            _d_zN += _r238;
        }
        {
            int _r225 = 0;
            int _r226 = 0;
            int _r227 = 0;
            int _r228 = 0;
            int _r229 = 0;
            int _r230 = 0;
            int _r231 = 0;
            get_field_value_pullback(y, 2, i, j, k - 1, xN, yN, zN, u_inlet, _d_w_km11, _d_y, &_r225, &_r226, &_r227, &_r228, &_r229, &_r230, &_r231);
            _d_i += _r226;
            _d_j += _r227;
            _d_k += _r228;
            _d_xN += _r229;
            _d_yN += _r230;
            _d_zN += _r231;
        }
        {
            int _r218 = 0;
            int _r219 = 0;
            int _r220 = 0;
            int _r221 = 0;
            int _r222 = 0;
            int _r223 = 0;
            int _r224 = 0;
            get_field_value_pullback(y, 2, i, j, k + 1, xN, yN, zN, u_inlet, _d_w_kp11, _d_y, &_r218, &_r219, &_r220, &_r221, &_r222, &_r223, &_r224);
            _d_i += _r219;
            _d_j += _r220;
            _d_k += _r221;
            _d_xN += _r222;
            _d_yN += _r223;
            _d_zN += _r224;
        }
        {
            int _r211 = 0;
            int _r212 = 0;
            int _r213 = 0;
            int _r214 = 0;
            int _r215 = 0;
            int _r216 = 0;
            int _r217 = 0;
            get_field_value_pullback(y, 2, i, j - 1, k, xN, yN, zN, u_inlet, _d_w_jm1, _d_y, &_r211, &_r212, &_r213, &_r214, &_r215, &_r216, &_r217);
            _d_i += _r212;
            _d_j += _r213;
            _d_k += _r214;
            _d_xN += _r215;
            _d_yN += _r216;
            _d_zN += _r217;
        }
        {
            int _r204 = 0;
            int _r205 = 0;
            int _r206 = 0;
            int _r207 = 0;
            int _r208 = 0;
            int _r209 = 0;
            int _r210 = 0;
            get_field_value_pullback(y, 2, i, j + 1, k, xN, yN, zN, u_inlet, _d_w_jp1, _d_y, &_r204, &_r205, &_r206, &_r207, &_r208, &_r209, &_r210);
            _d_i += _r205;
            _d_j += _r206;
            _d_k += _r207;
            _d_xN += _r208;
            _d_yN += _r209;
            _d_zN += _r210;
        }
        {
            int _r197 = 0;
            int _r198 = 0;
            int _r199 = 0;
            int _r200 = 0;
            int _r201 = 0;
            int _r202 = 0;
            int _r203 = 0;
            get_field_value_pullback(y, 2, i - 1, j, k, xN, yN, zN, u_inlet, _d_w_im1, _d_y, &_r197, &_r198, &_r199, &_r200, &_r201, &_r202, &_r203);
            _d_i += _r198;
            _d_j += _r199;
            _d_k += _r200;
            _d_xN += _r201;
            _d_yN += _r202;
            _d_zN += _r203;
        }
        {
            int _r190 = 0;
            int _r191 = 0;
            int _r192 = 0;
            int _r193 = 0;
            int _r194 = 0;
            int _r195 = 0;
            int _r196 = 0;
            get_field_value_pullback(y, 2, i + 1, j, k, xN, yN, zN, u_inlet, _d_w_ip1, _d_y, &_r190, &_r191, &_r192, &_r193, &_r194, &_r195, &_r196);
            _d_i += _r191;
            _d_j += _r192;
            _d_k += _r193;
            _d_xN += _r194;
            _d_yN += _r195;
            _d_zN += _r196;
        }
    } else if (_cond3) {
        {
            _d_cont += _d_result;
            _d_result = 0.F;
        }
        {
            _d_dy += _d_cont * (u_ip12 - u_im12) / 2. * dz;
            _d_dz += dy * _d_cont * (u_ip12 - u_im12) / 2.;
            _d_u_ip12 += (dy * dz / 2.) * _d_cont;
            _d_u_im12 += -(dy * dz / 2.) * _d_cont;
            _d_dx += _d_cont * (v_jp12 - v_jm12) / 2. * dz;
            _d_dz += dx * _d_cont * (v_jp12 - v_jm12) / 2.;
            _d_v_jp12 += (dx * dz / 2.) * _d_cont;
            _d_v_jm12 += -(dx * dz / 2.) * _d_cont;
            _d_dx += _d_cont * (w_kp12 - w_km12) / 2. * dy;
            _d_dy += dx * _d_cont * (w_kp12 - w_km12) / 2.;
            _d_w_kp12 += (dx * dy / 2.) * _d_cont;
            _d_w_km12 += -(dx * dy / 2.) * _d_cont;
        }
        {
            int _r320 = 0;
            int _r321 = 0;
            int _r322 = 0;
            int _r323 = 0;
            int _r324 = 0;
            int _r325 = 0;
            int _r326 = 0;
            get_field_value_pullback(y, 2, i, j, k - 1, xN, yN, zN, u_inlet, _d_w_km12, _d_y, &_r320, &_r321, &_r322, &_r323, &_r324, &_r325, &_r326);
            _d_i += _r321;
            _d_j += _r322;
            _d_k += _r323;
            _d_xN += _r324;
            _d_yN += _r325;
            _d_zN += _r326;
        }
        {
            int _r313 = 0;
            int _r314 = 0;
            int _r315 = 0;
            int _r316 = 0;
            int _r317 = 0;
            int _r318 = 0;
            int _r319 = 0;
            get_field_value_pullback(y, 2, i, j, k + 1, xN, yN, zN, u_inlet, _d_w_kp12, _d_y, &_r313, &_r314, &_r315, &_r316, &_r317, &_r318, &_r319);
            _d_i += _r314;
            _d_j += _r315;
            _d_k += _r316;
            _d_xN += _r317;
            _d_yN += _r318;
            _d_zN += _r319;
        }
        {
            int _r306 = 0;
            int _r307 = 0;
            int _r308 = 0;
            int _r309 = 0;
            int _r310 = 0;
            int _r311 = 0;
            int _r312 = 0;
            get_field_value_pullback(y, 1, i, j - 1, k, xN, yN, zN, u_inlet, _d_v_jm12, _d_y, &_r306, &_r307, &_r308, &_r309, &_r310, &_r311, &_r312);
            _d_i += _r307;
            _d_j += _r308;
            _d_k += _r309;
            _d_xN += _r310;
            _d_yN += _r311;
            _d_zN += _r312;
        }
        {
            int _r299 = 0;
            int _r300 = 0;
            int _r301 = 0;
            int _r302 = 0;
            int _r303 = 0;
            int _r304 = 0;
            int _r305 = 0;
            get_field_value_pullback(y, 1, i, j + 1, k, xN, yN, zN, u_inlet, _d_v_jp12, _d_y, &_r299, &_r300, &_r301, &_r302, &_r303, &_r304, &_r305);
            _d_i += _r300;
            _d_j += _r301;
            _d_k += _r302;
            _d_xN += _r303;
            _d_yN += _r304;
            _d_zN += _r305;
        }
        {
            int _r292 = 0;
            int _r293 = 0;
            int _r294 = 0;
            int _r295 = 0;
            int _r296 = 0;
            int _r297 = 0;
            int _r298 = 0;
            get_field_value_pullback(y, 0, i - 1, j, k, xN, yN, zN, u_inlet, _d_u_im12, _d_y, &_r292, &_r293, &_r294, &_r295, &_r296, &_r297, &_r298);
            _d_i += _r293;
            _d_j += _r294;
            _d_k += _r295;
            _d_xN += _r296;
            _d_yN += _r297;
            _d_zN += _r298;
        }
        {
            int _r285 = 0;
            int _r286 = 0;
            int _r287 = 0;
            int _r288 = 0;
            int _r289 = 0;
            int _r290 = 0;
            int _r291 = 0;
            get_field_value_pullback(y, 0, i + 1, j, k, xN, yN, zN, u_inlet, _d_u_ip12, _d_y, &_r285, &_r286, &_r287, &_r288, &_r289, &_r290, &_r291);
            _d_i += _r286;
            _d_j += _r287;
            _d_k += _r288;
            _d_xN += _r289;
            _d_yN += _r290;
            _d_zN += _r291;
        }
    }
}
