 oldimg = cv2.drawContours(oldimg, [box], 0, (0, 0, 255), 2)#在原图像上画出矩形,TODO:正式识别时记得删除
                cv2.namedWindow("edge4", cv2.WINDOW_NORMAL)
                cv2.imshow("edge4", oldimg)
                cv2.waitKey()
                cv2.destroyAllWindows()