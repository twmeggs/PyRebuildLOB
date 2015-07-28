"""
Module to provide functions which
can be used to reconstruct an orderbook
from Level 1 tick data.

A simulation of limit order execution
is run when running as main.
"""

import pandas as pd
import numpy as np
import datetime as dt
import random

def df_from_ticks(file_location='./sample_ticks.csv', index_col='Date and Time'):
    """Return a dataframe from tick data."""
    raw_tick_df = pd.read_csv(file_location, index_col=index_col,
                              parse_dates=True, infer_datetime_format=True)
    return raw_tick_df


def increase_granularity(df_from_ticks):
    """Return re-indexed dataframe with arbitrary microsecond timestamps."""
    index_as_int = df_from_ticks.index.astype(np.int64)
    arbitrary_microseconds = []
    place_within_second = 1
    nano_to_micro = 1000


    for tick in np.arange(1,len(index_as_int)):
        if tick == 1:
            arbitrary_microseconds.append(place_within_second * nano_to_micro)
            if index_as_int[tick] != index_as_int[tick-1]:
                arbitrary_microseconds.append(place_within_second * nano_to_micro)
            else:
                place_within_second += 1
                arbitrary_microseconds.append(place_within_second * nano_to_micro)
        else:
            if index_as_int[tick] != index_as_int[tick-1]:
                place_within_second = 1
                arbitrary_microseconds.append(place_within_second * nano_to_micro)
            else:
                place_within_second += 1
                arbitrary_microseconds.append(place_within_second * nano_to_micro)


    new_idx_int = index_as_int + arbitrary_microseconds
    new_idx = pd.DatetimeIndex(data=new_idx_int)
    new_df = pd.DataFrame(index=new_idx, data=df_from_ticks.as_matrix(), \
                          columns=(['side','price','size']))

    return new_df


def build_book(microsecond_ticks):
    df_columns = ('curr_bid', 'curr_ask', 'curr_bid_size', 'curr_ask_size', \
                  'trade', 'trade_prc', 'trade_size', 'last_trade_prc', \
                  'last_trade_side')
    book_df = pd.DataFrame(index=microsecond_ticks.index, columns=df_columns)

    print 'starting book build'
    for tick in np.arange(len(microsecond_ticks.index)):

        if tick%10000 == 0:
            print 'still building book, row ', tick, 'of ', len(microsecond_ticks)
        if tick == 0:
            # collect variables to persist
            bid = microsecond_ticks.price[tick]
            ask = microsecond_ticks.price[tick]
            bid_size = microsecond_ticks.size[tick]
            ask_size = microsecond_ticks.size[tick]

            # update first row of book
            book_df.curr_bid[tick] = bid
            book_df.curr_ask[tick] = ask
            book_df.curr_bid_size[tick] = bid_size
            book_df.curr_ask_size[tick] = ask_size
            book_df.last_trade_side[tick] = 'tbd'

            if microsecond_ticks.side[tick] == 'TRADE':
                # collect variables
                last = microsecond_ticks.price[tick]
                last_size = microsecond_ticks.size[tick]

                book_df.trade[tick] = True
                book_df.trade_prc[tick] = last
                book_df.trade_size[tick] = last_size
                book_df.last_trade_prc[tick] = last

            else:
                # collect variables
                last = 0.
                book_df.trade[tick] = False
                book_df.trade_prc[tick] = 0.
                book_df.trade_size[tick] = 0.
                book_df.last_trade_prc[tick] = 0.

        else:

            if microsecond_ticks.side[tick] == 'BID':
                # collect variable to persist
                bid = microsecond_ticks.price[tick]
                bid_size = microsecond_ticks.size[tick]

                # update book
                book_df.curr_bid[tick] = bid
                book_df.curr_ask[tick] = ask
                book_df.curr_bid_size[tick] = bid_size
                book_df.curr_ask_size[tick] = ask_size
                book_df.trade[tick] = False
                book_df.trade_prc[tick] = 0.
                book_df.trade_size[tick] = 0.
                book_df.last_trade_prc[tick] = last
                book_df.last_trade_side[tick] = book_df.last_trade_side[tick-1]


            elif microsecond_ticks.side[tick] == 'ASK':
                # collect variable to persist
                ask = microsecond_ticks.price[tick]
                ask_size = microsecond_ticks.size[tick]

                # update book
                book_df.curr_bid[tick] = bid
                book_df.curr_ask[tick] = ask
                book_df.curr_bid_size[tick] = bid_size
                book_df.curr_ask_size[tick] = ask_size
                book_df.trade[tick] = False
                book_df.trade_prc[tick] = 0.
                book_df.trade_size[tick] = 0.
                book_df.last_trade_prc[tick] = last
                book_df.last_trade_side[tick] = book_df.last_trade_side[tick-1]

            elif microsecond_ticks.side[tick] == 'TRADE':
                # collect variable to persist
                last = microsecond_ticks.price[tick]
                last_size = microsecond_ticks.size[tick]

                # update book
                book_df.curr_bid[tick] = bid
                book_df.curr_ask[tick] = ask
                book_df.curr_bid_size[tick] = bid_size
                book_df.curr_ask_size[tick] = ask_size
                book_df.trade[tick] = True
                book_df.trade_prc[tick] = last
                book_df.trade_size[tick] = last_size
                book_df.last_trade_prc[tick] = last
                book_df.last_trade_side[tick] = book_df.last_trade_side[tick-1]

                # determine side of trade and queue depletion
                if microsecond_ticks.price[tick] == book_df.curr_bid[tick]:
                    book_df.last_trade_side[tick] = 'BID'
                    # remove traded amount from queue amount
                    book_df.curr_bid_size[tick] = (book_df.curr_bid_size[tick] -
                                                   book_df.trade_size[tick])

                elif microsecond_ticks.price[tick] == book_df.curr_ask[tick]:
                    book_df.last_trade_side[tick] = 'ASK'
                    # remove traded amount from queue amount
                    book_df.curr_ask_size[tick] = (book_df.curr_ask_size[tick] -
                                                   book_df.trade_size[tick])

                else:
                    book_df.last_trade_side[tick] = 'tbd'

                # handle consecutive trades effect on bid/ask prices
                if book_df.trade[tick-1]:
                    if (book_df.last_trade_side[tick-1] == "ASK") and \
                        (microsecond_ticks.price[tick] > book_df.curr_ask[tick]):

                        book_df.last_trade_side[tick] = "ASK"
                        book_df.curr_ask[tick] = microsecond_ticks.price[tick]
                        book_df.curr_ask_size[tick] = book_df.curr_ask_size[tick-1]

                    elif (book_df.last_trade_side[tick-1] == "BID") and \
                        (microsecond_ticks.price[tick] < book_df.curr_bid[tick]):

                        book_df.last_trade_side[tick] = "BID"
                        book_df.curr_bid[tick] = microsecond_ticks.price[tick]
                        book_df.curr_bid_size[tick] = book_df.curr_bid_size[tick-1]


    # remove first n rows where the book is first being built
    ts = book_df[(book_df.curr_bid != book_df.curr_ask) &
         (book_df.last_trade_prc != 0.)].index[0]
    n = np.where(book_df.index == ts)[0][0]
    book_df = book_df.drop(book_df.index[:n])

    return book_df


def check_order(direction, limit_prc, queue_pos, book_state):
    filled = False
    fill_type = 'not_filled'

    if direction == 1:

        if book_state.curr_ask == limit_prc:
            filled = True
            fill_type = 'ask_price_changed'

        elif book_state.trade and (book_state.trade_prc == book_state.curr_bid):
            queue_pos = queue_pos - book_state.trade_size
            if queue_pos < 0:
                filled = True
                fill_type = 'queue_position_exhausted'

    elif direction == 0:

        if book_state.curr_bid == limit_prc:
            filled = True
            fill_type = 'bid_price-changed'

        elif book_state.trade and (book_state.trade_prc == book_state.curr_ask):
            queue_pos = queue_pos - book_state.trade_size
            if queue_pos < 0:
                filled = True
                fill_type = 'queue_position_exhausted'

    return filled, queue_pos, fill_type


def simulate_limit_orders(book, execution_work_time=30):

    entry_time = book.index[0]
    last_tick = book.index[-1]
    order_close_time = book.index[-1]
    num_orders_raised = 0.
    num_orders_filled = 0.
    state = 'no_order_live'

    for tick in book.index:
        print 'limit orders raised:', num_orders_raised, 'orders filled:', num_orders_filled
        if state == 'working_order':

            if tick >= order_close_time:
                state = 'no_order_live'

            else:

                filled, queue_pos, fill_type = check_order(buy_or_sell, limit_prc, queue_pos, book.ix[tick])

                if filled:
                    num_orders_filled += 1
                    state = 'no_order_live'

        elif state == 'no_order_live':

            ticks_ahead = random.randrange(0,500)
            current_tick_loc = book.index.get_loc(tick)
            entry_time = book.index[min(ticks_ahead + current_tick_loc,len(book.index) - 1)]
            buy_or_sell = random.randint(0,1)
            state = 'waiting_to_raise_order'

        elif state == 'waiting_to_raise_order':

            if tick == entry_time:
                # avoid ticks in the book that are 'trades'
                if book.trade[tick]:

                    state = 'no_order_live'

                else:

                    if buy_or_sell == 0:

                        limit_prc = book.curr_ask[tick]
                        queue_pos = book.curr_ask[tick] + 1.
                        num_orders_raised += 1
                        order_close_time = tick + dt.timedelta(seconds=execution_work_time)
                        state = 'working_order'

                    else:

                        limit_prc = book.curr_bid[tick]
                        queue_pos = book.curr_bid[tick] + 1.
                        num_orders_raised += 1
                        order_close_time = tick + dt.timedelta(seconds=execution_work_time)
                        state = 'working_order'

    return 1

def trade_intensity(day_LOB):
    """Return a minutely dataframe with sum of market order events"""
    trade_events = day_LOB.trade[day_LOB.trade==True]
    trades_per_min = trade_events.resample('1min', how='count')
    tmp_idx = trades_per_min.index.time
    trades_per_min = pd.Series(data=trades_per_min.values, index=tmp_idx)
    return trades_per_min

def mean_trade_size(day_LOB):
    """Return minutely dataframe with mean trade size of market order events"""
    trade_sizes = day_LOB.trade_size[day_LOB.trade==True]
    trade_sizes = trade_sizes.astype(int)
    mean_trade_size = trade_sizes.resample('1min', how='mean')
    tmp_idx = mean_trade_size.index.time
    mean_trade_size = pd.Series(data=mean_trade_size.values, index=tmp_idx)
    return mean_trade_size


if __name__ == "__main__":
    ticks = df_from_ticks()
    granular_ticks = increase_granularity(ticks)
    order_book = build_book(granular_ticks)
    simulate_limit_orders(order_book)