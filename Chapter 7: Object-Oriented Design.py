#7.1 Deck of Cards

from enum import Enum
import random

class Sign(Enum):
    Livce = 1
    Baklava = 2
    Detelina = 3
    Srce = 4

class Card:
    def __init__(self, sign, number):
        self.sign = sign
        self.number = number
        self.available = True

    def __eq__(self, other):
        return self.sign == other.sign and self.number == other.number

    def is_available(self):
        return self.available

    def mark_unavailable(self):
        self.available = False

    def mark_uavailable(self):
        self.available = True

    def __str__(self):
        return "Number: {0}, Sign: {1}".format(self.number, self.sign.name)

class DeckOfCards:

    _all_cards = [Card(sign, num) for sign in Sign for num in range(1, 15)]

    def __init__(self):
        new_deck = DeckOfCards._all_cards.copy()
        random.shuffle(new_deck)
        self.deck = deque(new_deck)
        self.used = []

    def shuffle(self):
        random.shuffle(self.deck)

    def deal(self):
        card = self.deck.pop()
        print(card)
        used.append(card)

class Hand:
    def __init__(self):
        self.cards = []

    def add_card(self, card):
        self.cards.append(card)

    def score(self):
        score = 0
        for card in self.cards:
            score += card.value
        return score

#7.2 Call Center
class Employee:
    def __init__(self, embg, name):
        self.embg = embg
        self.name = name
        self.available = True

    def free(self):
        self.available = True

    def make_unavailable(self):
        self.available = False

class Respondent(Employee):
    def __init__(self, embg, name, department):
        super.__init__(embg, name, department)
        self.manager = department.manager

class Manager(Employee):
    def __init__(self, embg, name, department):
        super.__init__(embg, name, department)

class Director(Employee):
    def __init__(self, embg, name, departments = []):
        super.__init__(embg, name, departments)

import time

class CallStatus(Enum):
    pending = 1
    active = 2
    finished = 3

import uuid

class Call:
    def __init__(self):
        self.status = CallStatus.pending
        self.employee = None
        self.duration = -1
        self.id == uuid.uuid4()

    def anwer(self, employee):
        self.employee = employee
        self.status = CallStatus.active
        self.start = time.time()

    def end(self):
        self.duration = (time.time() - self.start)
        self.status = CallStatus.finished
        self.employee.free()

    def escalate(self, employee):
        employee.make_unavailable()
        self.employee = employee

    def __eq__(self, other):
        return self.id == other.id

class CallCenter:
    def __init__(self):
        self.employees = []
        self.finished_calls = []
        self.active_calls = []
        self.pending_calls = deque()

    def call(self, call):
        self.pending_calls.append(call)

    def answer_call(self):
        call = self.pending_calls.popleft()

        for employee in self.employees:
            if issubclass(employee, Respondent):
                if employee.available == True:
                    employee.available = False
                    call.anwer(employee)
                    self.active_calls.append(call)
                    break

        if call.status == CallStatus.pending:
            print("no free employees")

    def end_call(self, call):
        for call_ in self.active_calls:
            if call_ == call:
                self.active_calls.remove(call_)
                call_.end()
                call_.employee.available = True
                self.finished_calls.append(call_)
                return call_
        return None

    def escalate_call(self, call):
        if call.status != CallStatus.active:
            raise Exception("Call is not active")

        manager = call.employee.manager
        if manager.available == True:
            call.escalate(manager)
            return True
        elif manager.director.available == True:
            call.escalate(manager.director)
            return True

        return False
