#!/usr/bin/env python3

import requests
from streamdeck_sdk import (
    StreamDeck,
    Action,
    events_received_objs,
    events_sent_objs,
    image_bytes_to_base64,
    image_file_to_base64,
    logger,
)
import logging
import os
import sqlite3


def current_experiment():
    cursor = conn.cursor()
    cursor.execute("select real_experiment_id from current_experiment")
    row = cursor.fetchone()
    if row is None:
        logger.warn("Cannot find the current experiment ID")
        # Hmm, what to switch to? Ideally, the bmap demo starter?
        cursor.close()
        return None
    current_experiment_id = row[0]
    cursor.close()
    return current_experiment_id

def get_current_value(column):
    cursor = conn.cursor()
    current_experiment_id = current_experiment()
    cursor.execute(f"select {column} from real_experiments where real_experiment_id = ?",
                   [current_experiment_id])
    row = cursor.fetchone()
    if row is None:
        logger.warn(f"Cannot find a corresponding real experiment to {current_experiment_id}")
        cursor.close()
        return None
    value = row[0]
    cursor.close()
    return value

def state_from_value(column):
    result = get_current_value(column)
    logger.warn(f"The current value of {column} is {result}")
    if result == 0:
        return 0
    elif result == 1:
        return 1
    logging.warn(f"The value of {column} was {result}, which cannot be displayed.")
    return None

def state_from_comparison(column1, column2):
    v1 = get_current_value(column1)
    v2 = get_current_value(column2)
    logging.warn(f"The value of {column1} is {v1}. The value of {column2} is {v2}. {v1==v2=}")
    if v1 == v2:
        return 1
    return 0

def state_from_tri_comparison(column2, column3):
    v1 = get_current_value('ground_truth')
    v2 = get_current_value(column2)
    v3 = get_current_value(column3)
    logging.warn(f"The value of ground_truth is {v1}. The value of {column2} is {v2}. {v1==v2=}")
    logging.warn(f"The value of ground_truth is {v1}. The value of {column3} is {v3}. {v1==v3=}")
    if v1 != v2 or v1 != v3:
        return 1
    return 0

class WhatWelchThinks(Action):
    UUID = "au.org.ifost.superwelchdemo.what-welch-thinks"

    def on_will_appear(self, obj):
        self.set_state(
            context=obj.context,
            state=state_from_value('welch_ttest_result')
        )

class WhatBMAP1Thinks(Action):
    UUID = "au.org.ifost.superwelchdemo.what-bmap1-thinks"
    
    def on_will_appear(self, obj):
        self.set_state(
            context=obj.context,
            state=state_from_value('bmap1_result')
        )

class WhatBMAP3Thinks(Action):
    UUID = "au.org.ifost.superwelchdemo.what-bmap3-thinks"

    def on_will_appear(self, obj):
        self.set_state(
            context=obj.context,
            state=state_from_value('bmap3_result')
        )    
    
class EatCountVsWelchBmap1(Action):
    UUID = "au.org.ifost.superwelchdemo.eat-count-vs-welch-bmap1"
    
    def on_will_appear(self, obj):
        self.set_state(
            context=obj.context,
            state=state_from_tri_comparison('welch_ttest_result', 'bmap1_result')
        )        

class EatCountVsWelchBmap3(Action):
    UUID = "au.org.ifost.superwelchdemo.eat-count-vs-welch-bmap3"

    def on_will_appear(self, obj):
        self.set_state(
            context=obj.context,
            state=state_from_tri_comparison('welch_ttest_result', 'bmap3_result')
        )           
    
class WelchCount(Action):
    UUID = "au.org.ifost.superwelchdemo.welch-count"

    def on_will_appear(self, obj):
        self.set_state(
            context=obj.context,
            state=state_from_comparison('ground_truth', 'welch_ttest_result')
        )               

    
class Bmap1Count(Action):
    UUID = "au.org.ifost.superwelchdemo.bmap1-count"

    def on_will_appear(self, obj):
        self.set_state(
            context=obj.context,
            state=state_from_comparison('ground_truth', 'bmap1_result')
        )                   

class Bmap3Count(Action):
    UUID = "au.org.ifost.superwelchdemo.bmap3-count"

    def on_will_appear(self, obj):
        self.set_state(
            context=obj.context,
            state=state_from_comparison('ground_truth', 'bmap3_result')
        )                       
    
class JumpToLollyPage(Action):
    UUID = "au.org.ifost.superwelchdemo.lolly-jump"

    def on_key_down(self, obj: events_received_objs.KeyDown):
        if row is None:
            logger.warn(f"Cannot find a corresponding real experiment to {current_experiment_id}")
            cursor.close()
            return
        welch_result = row[0]
        bmap3_result = row[1]
        ground_truth = row[2]
        welch_profile = "C" if welch_result == ground_truth else "W"
        bmap3_profile = "C" if bmap3_result == ground_truth else "W"
        profile_name = f"W{welch_profile}B{bmap3_profile}"
        logger.info(f"{welch_result=}, {bmap3_result=}, {ground_truth=}")
        logger.info(f"Switching to profile {profile_name}")
        self.switch_to_profile(device=obj.device, profile=profile_name)
        cursor.close()


class JumpToTestResultProfile(Action):
    UUID = "au.org.ifost.superwelchdemo.test-results"

    def on_key_down(self, obj: events_received_objs.KeyDown):
        cursor = conn.cursor()
        cursor.execute("select real_experiment_id from current_experiment")
        row = cursor.fetchone()
        if row is None:
            logger.warn("Cannot find the current experiment ID")
            # Hmm, what to switch to? Ideally, the bmap demo starter?
            cursor.close()
            return
        current_experiment_id = row[0]
        cursor.execute("select welch_ttest_result, bmap3_result from real_experiments where real_experiment_id = ?", [current_experiment_id])
        row = cursor.fetchone()
        if row is None:
            logger.warn(f"Cannot find a corresponding real experiment to {current_experiment_id}")
            cursor.close()
            return
        welch_result = row[0]
        bmap3_result = row[1]
        logger.info(f"{welch_result=}")
        logger.info(f"{bmap3_result=}")

        welch_profile = "Y" if welch_result == 1 else "N"
        bmap3_profile = "Y" if bmap3_result == 1 else "N"
        profile_name = f"W{welch_profile}B{bmap3_profile}"
        logger.info(f"Switching to {profile_name}")
        
        self.switch_to_profile(device=obj.device, profile=profile_name)
        cursor.close()

if __name__ == '__main__':
    session_file = os.path.expanduser("~/.bmap-demo-session.sqlite")
    logger.info(session_file)
    conn = sqlite3.connect(session_file)

    StreamDeck(
        actions=[
            JumpToTestResultProfile(),
            JumpToLollyPage(),
            WhatWelchThinks(),
            WhatBMAP1Thinks(),
            WhatBMAP3Thinks(),
            EatCountVsWelchBmap1(),
            EatCountVsWelchBmap3(),
            WelchCount(),
            Bmap1Count(),
            Bmap3Count()
        ],
        log_file='/tmp/demo.log',
        log_level=logging.DEBUG,
        log_backup_count=1,
    ).run()
