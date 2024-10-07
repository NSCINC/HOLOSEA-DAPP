void invest(InvestmentContract *contract, const char *investor, long amount) {
    if (amount <= 0) {
        printf("Investment amount must be greater than zero\n");
        return;
    }

    int investorIndex = -1;

    for (int i = 0; i < contract->investorCount; i++) {
        if (strcmp(contract->investors[i], investor) == 0) {
            investorIndex = i;
            break;
        }
    }

    if (investorIndex == -1) {
        printf("Investor not found\n");
        return;
    }

    if (!contract->authorizedInvestors[investorIndex]) {
        printf("Investor is not authorized\n");
        return;
    }

    if (amount > contract->balances[investorIndex]) {
        printf("Insufficient balance\n");
        return;
    }

    contract->balances[investorIndex] -= amount;
    contract->investedAmount[investorIndex] += amount;
    printf("Investment of %ld made by %s\n", amount, investor);
}
