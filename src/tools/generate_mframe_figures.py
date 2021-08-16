import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':

    m_frames = [3, 5, 10, 15, 20, 25, 30, 35]
    # mota_kalman = np.asarray([0.335360, 0.336384, 0.341358, 0.342194, 0.340877, 0.341107, 0.341964]) * 100
    # mota_lstm = np.asarray([0.338411, 0.341755, 0.345475,  0.346185, 0.346352, 0.346311, 0.346248]) * 100

    mota_kalman = np.asarray([0.395106, 0.397508, 0.399817, 0.400358, 0.400202, 0.400223, 0.400316, 0.400503]) * 100
    mota_lstm = np.asarray([0.400576, 0.402448, 0.406088, 0.406982, 0.407180, 0.407315, 0.407388, 0.407429]) * 100

    plt.title('m_frames MOTA')
    plt.plot(m_frames, mota_kalman, label='Kalman')
    plt.plot(m_frames, mota_lstm, label='LSTM')
    plt.legend()
    plt.xlabel('m_frame')
    plt.ylabel('MOTA (%)')
    plt.show()