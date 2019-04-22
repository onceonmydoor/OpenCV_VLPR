xl, xr, yh, yl = self.accurate_place(card_img_hsv, limit1, limit2, color)
                # if yl == yh and xl == xr:
                #     continue
                # need_accurate = False
                # if yl >= yh:
                #     yl = 0
                #     yh = row_num
                #     need_accurate = True
                # if xl >= xr:
                #     xl = 0
                #     xr = col_num
                #     need_accurate = True
                # card_imgs[card_index] = card_img[yl:yh, xl:xr] if color !="green" or yl < (yh - yl) // 4 else card_img[yl - (yh - yl) // 4:yh,xl:xr]