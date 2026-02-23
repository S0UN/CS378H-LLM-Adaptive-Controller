from datasets import load_dataset, DatasetDict


class DatasetService:
    # Supported datasets: shortcut name -> HuggingFace repo ID
    url_map: dict[str, str] = {
        "Capybara":       "LDJnr/Capybara",
        "OpenHermes-2.5": "teknium/OpenHermes-2.5",
    }

    # In-memory cache so the same dataset isn't loaded twice
    dataset_map: dict[str, DatasetDict] = {}

    def get_database(self, dataset_name: str) -> DatasetDict:
        """
        Load a dataset by shortcut name. Returns a cached copy if already loaded.

        Args:
            dataset_name: Key from url_map (e.g. "Capybara" or "OpenHermes-2.5")

        Returns:
            HuggingFace DatasetDict (e.g. {"train": Dataset(...)})
        """
        if dataset_name in self.dataset_map:
            return self.dataset_map[dataset_name]

        url = self.url_map[dataset_name]
        dataset: DatasetDict = load_dataset(url)

        self.dataset_map[dataset_name] = dataset
        return dataset
